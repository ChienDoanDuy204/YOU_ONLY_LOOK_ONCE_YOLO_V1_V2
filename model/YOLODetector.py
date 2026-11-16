import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from model.Kmean_YOLOv2 import *
from torch.utils.data import DataLoader
from src.data.orgainize_data import collate_fn
from functools import partial
from tqdm import tqdm
from torch.amp import GradScaler
from torchvision.ops import batched_nms

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        MeanAveragePrecision = None


class DarkNet(nn.Module):
    """Backbone feature extractor based on pre-trained ResNet-18 without pooling/fc layers."""

    def __init__(self) -> None:
        super().__init__()
        self.extractor = timm.create_model("resnet18.a3_in1k", pretrained=True)
        # Bỏ lớp AVG_POOLING và FC của ResNet
        self.extractor.global_pool = nn.Identity()
        self.extractor.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)
        return x


class YOLOv2(nn.Module):
    """YOLOv2 object detector with ResNet-18 backbone, anchor-based head, and mAP evaluation."""

    def __init__(self, num_classes: int = 20) -> None:
        super().__init__()
        self.backbone = DarkNet()
        self.num_classes = num_classes
        self.batch_norm1 = nn.BatchNorm2d(
            num_features=512,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.layer_conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batch_norm2 = nn.BatchNorm2d(
            num_features=1024,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.layer_conv2 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batch_norm3 = nn.BatchNorm2d(
            num_features=1024,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.layer_conv3 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batch_norm4 = nn.BatchNorm2d(
            num_features=1024,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.layer_conv4 = nn.Conv2d(
            in_channels=1024,
            out_channels=5 * (5 + num_classes),
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass through DarkNet backbone and detection convolutional head."""
        ## Convolution layer 0
        feature_maps = self.backbone(x)
        feature_maps = self.batch_norm1(feature_maps)
        feature_maps = F.relu(feature_maps, inplace=True)

        ## Convolution layer 1
        feature_maps = self.layer_conv1(feature_maps)
        feature_maps = self.batch_norm2(feature_maps)
        feature_maps = F.relu(feature_maps, inplace=True)

        ## Convolution layer 2
        feature_maps = self.layer_conv2(feature_maps)
        feature_maps = self.batch_norm3(feature_maps)
        feature_maps = F.relu(feature_maps, inplace=True)

        ## Convoltuion layer 3
        feature_maps = self.layer_conv3(feature_maps)
        feature_maps = self.batch_norm4(feature_maps)
        feature_maps = F.dropout(feature_maps, p=0.3, training=self.training)
        feature_maps = F.relu(feature_maps, inplace=True)

        ## Head prediction
        output = self.layer_conv4(feature_maps)
        return output # shape (B, 5*(5+num_classes), H/32, W/32) where 5*(5+num_classes) is 5*(bx,by,sqrt(bw),sqrt(bh), 1 confidence) + 5*numclass

    def find_best_anchor_shape(
        self,
        dataset: torch.utils.data.Dataset,
        num_anchors: int = 5,
        max_iterators: int = 100,
    ) -> torch.Tensor:
        """Find popular bounding box shapes in GT annotations using K-Means clustering.

        Parameters:
        -----------
        dataset : torch.utils.data.Dataset
            Dataset whose bounding box dimensions will be clustered.
        num_anchors : int, default=5
            Number of anchor clusters to generate.
        max_iterators : int, default=100
            Maximum number of K-Means clustering iterations.

        Returns:
        --------
        torch.Tensor
            Tensor of shape (num_anchors, 2) containing sorted (width, height) anchor dimensions.
        """
        data_cluster = dataset[0][1]["boxes"].clone()[:, 2:]
        for i in range(1, dataset.__len__()):
            data_cluster = torch.cat([data_cluster, dataset[i][1]["boxes"].clone()[:, 2:]], dim=0)
        Kmean_cluster = Kmean(num_anchors=num_anchors, max_iterators=max_iterators)
        anchor = Kmean_cluster(data_cluster)
        return anchor

    def build_target(
        self,
        anchors: torch.Tensor,
        img_size: tuple[int, int] = (416, 416),
        target_padded: torch.Tensor | None = None,
        maskGTboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build YOLOv2 target tensor of shape (B, 5*(5+num_classes), H//32, W//32).

        Parameters:
        -----------
        anchors : torch.Tensor
            Anchor box dimensions tensor of shape (5, 2).
        img_size : tuple[int, int], default=(416, 416)
            Target image resolution (height, width) for grid calculation.
        target_padded : torch.Tensor | None, default=None
            Batch of padded targets of shape (B, M, 5) with [cx, cy, w, h, label].
        maskGTboxes : torch.Tensor | None, default=None
            Boolean mask of shape (B, M) indicating valid ground truth boxes.

        Returns:
        --------
        torch.Tensor
            Target tensor of shape (B, 5*(5+num_classes), H//32, W//32).
        """
        if target_padded is None:
            raise ValueError("target_padded must not be None")
        if target_padded.ndim != 3 or target_padded.shape[-1] != 5:
            raise ValueError(f"Expected target_padded shape (B, M, 5), got {tuple(target_padded.shape)}")

        stride = 32
        grid_h = img_size[0] // stride
        grid_w = img_size[1] // stride
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"img_size {img_size} must be at least ({stride}, {stride})")

        num_anchors = 5
        anchors = anchors.to(device=target_padded.device, dtype=target_padded.dtype)
        if anchors.shape != (num_anchors, 2):
            raise ValueError(f"Expected anchors shape ({num_anchors}, 2), got {tuple(anchors.shape)}")

        batch_size, max_boxes, _ = target_padded.shape
        target = torch.zeros(
            (batch_size, grid_h, grid_w, num_anchors, 5 + self.num_classes),
            device=target_padded.device,
            dtype=target_padded.dtype,
        )

        gt_wh = target_padded[..., 2:4]
        valid_mask = (gt_wh[..., 0] > 0) & (gt_wh[..., 1] > 0)
        if maskGTboxes is not None:
            if maskGTboxes.shape != (batch_size, max_boxes):
                raise ValueError(
                    f"Expected maskGTboxes shape {(batch_size, max_boxes)}, got {tuple(maskGTboxes.shape)}"
                )
            valid_mask = valid_mask & maskGTboxes.to(device=target_padded.device, dtype=torch.bool)

        # Centered shape IoU between GT boxes (B, M, 2) and anchors (5, 2)
        inter_wh = torch.minimum(gt_wh.unsqueeze(2), anchors.view(1, 1, num_anchors, 2))
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        gt_area = gt_wh[..., 0] * gt_wh[..., 1]
        anchor_area = anchors[:, 0] * anchors[:, 1]
        union_area = gt_area.unsqueeze(2) + anchor_area.view(1, 1, num_anchors) - inter_area
        ious = inter_area / union_area.clamp(min=1e-6)
        anchor_order = torch.argsort(ious, dim=-1, descending=True)

        # Grid indices (x -> col, y -> row) and cell-relative offsets in [0, 1]
        gx = target_padded[..., 0] / stride
        gy = target_padded[..., 1] / stride
        cols = torch.floor(gx).long().clamp(0, grid_w - 1)
        rows = torch.floor(gy).long().clamp(0, grid_h - 1)
        bx = (gx - cols.to(target_padded.dtype)).clamp(0.0, 1.0)
        by = (gy - rows.to(target_padded.dtype)).clamp(0.0, 1.0)
        cls_ids = target_padded[..., 4].long()

        valid_b, valid_m = torch.nonzero(valid_mask, as_tuple=True)
        for b_idx, m_idx in zip(valid_b.tolist(), valid_m.tolist()):
            cls_idx = int(cls_ids[b_idx, m_idx].item())
            if not (0 <= cls_idx < self.num_classes):
                raise ValueError(f"Class index {cls_idx} out of range [0, {self.num_classes - 1}]")

            row = int(rows[b_idx, m_idx].item())
            col = int(cols[b_idx, m_idx].item())

            candidates = anchor_order[b_idx, m_idx].tolist()
            chosen_anchor = next(
                (a for a in candidates if target[b_idx, row, col, a, 4] == 0.0),
                candidates[0],
            )

            aw = anchors[chosen_anchor, 0].clamp(min=1e-6)
            ah = anchors[chosen_anchor, 1].clamp(min=1e-6)
            tw = gt_wh[b_idx, m_idx, 0] / aw
            th = gt_wh[b_idx, m_idx, 1] / ah

            target[b_idx, row, col, chosen_anchor, 0] = bx[b_idx, m_idx]
            target[b_idx, row, col, chosen_anchor, 1] = by[b_idx, m_idx]
            target[b_idx, row, col, chosen_anchor, 2] = tw
            target[b_idx, row, col, chosen_anchor, 3] = th
            target[b_idx, row, col, chosen_anchor, 4] = 1.0
            target[b_idx, row, col, chosen_anchor, 5:] = 0.0
            target[b_idx, row, col, chosen_anchor, 5 + cls_idx] = 1.0

        # Permute (B, H, W, 5, 5+C) -> (B, 5, 5+C, H, W) -> (B, 5*(5+C), H, W)
        return target.permute(0, 3, 4, 1, 2).reshape(
            batch_size, num_anchors * (5 + self.num_classes), grid_h, grid_w
        )

    def _convert_gt_to_metric_format(
        self,
        target_scaled: torch.Tensor,
        mask_gt_boxes: torch.Tensor,
        target_size: int,
    ) -> list[dict[str, torch.Tensor]]:
        """Convert padded GT targets into list of dicts required by MeanAveragePrecision.

        Parameters:
        -----------
        target_scaled : torch.Tensor
            Tensor of shape (B, M, 5) containing ground truth bounding boxes and class labels
            scaled to current target_size, where [..., :4] represents [cx, cy, w, h] and
            [..., 4] represents the integer class label.
        mask_gt_boxes : torch.Tensor
            Boolean tensor of shape (B, M) indicating whether a box slot is a valid ground
            truth annotation (True) or zero-padding (False).
        target_size : int
            Integer image resolution (width and height in pixels, e.g., 416) used to clamp
            converted bounding box coordinates to image boundaries.

        Returns:
        --------
        list[dict[str, torch.Tensor]]
            A list of length B where each dictionary contains:
                - 'boxes': FloatTensor of shape (num_gt, 4) in [x1, y1, x2, y2] format on CPU.
                - 'labels': LongTensor of shape (num_gt,) containing class IDs on CPU.
        """
        batch_size = target_scaled.shape[0]
        target_list: list[dict[str, torch.Tensor]] = []
        for b in range(batch_size):
            valid = mask_gt_boxes[b] & (target_scaled[b, :, 2] > 0) & (target_scaled[b, :, 3] > 0)
            boxes = target_scaled[b, valid, :4]
            labels = target_scaled[b, valid, 4].long()
            if boxes.numel() > 0:
                cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = (cx - 0.5 * w).clamp(min=0.0, max=float(target_size))
                y1 = (cy - 0.5 * h).clamp(min=0.0, max=float(target_size))
                x2 = (cx + 0.5 * w).clamp(min=0.0, max=float(target_size))
                y2 = (cy + 0.5 * h).clamp(min=0.0, max=float(target_size))
                xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
            else:
                xyxy = torch.zeros((0, 4), dtype=torch.float32, device=target_scaled.device)
                labels = torch.zeros((0,), dtype=torch.long, device=target_scaled.device)
            target_list.append({"boxes": xyxy.detach().cpu(), "labels": labels.detach().cpu()})
        return target_list

    def _prepare_batch(
        self,
        imgs: torch.Tensor,
        target_padded: torch.Tensor,
        mask_gt_boxes: torch.Tensor,
        anchors: torch.Tensor,
        target_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resize image batch to (target_size, target_size), scale boxes, and build target tensor.

        Parameters:
        -----------
        imgs : torch.Tensor
            Batch of input images of shape (B, 3, in_h, in_w).
        target_padded : torch.Tensor
            Padded targets tensor of shape (B, M, 5) with [cx, cy, w, h, label].
        mask_gt_boxes : torch.Tensor
            Boolean mask of shape (B, M) indicating valid ground truth annotations.
        anchors : torch.Tensor
            Base anchor dimensions tensor of shape (5, 2).
        target_size : int
            Square image dimension to which imgs and boxes are scaled.

        Returns:
        --------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - imgs_resized: Resized image tensor of shape (B, 3, target_size, target_size).
            - targets: Target grid tensor of shape (B, 5*(5+num_classes), target_size//32, target_size//32).
            - target_scaled: Scaled bounding box tensor of shape (B, M, 5) with [cx, cy, w, h, label].
        """
        _, _, in_h, in_w = imgs.shape
        if (in_h, in_w) != (target_size, target_size):
            imgs = F.interpolate(imgs, size=(target_size, target_size), mode="bilinear", align_corners=False)
            target_scaled = target_padded.clone()
            scale_x = target_size / float(in_w)
            scale_y = target_size / float(in_h)
            target_scaled[..., 0] *= scale_x
            target_scaled[..., 2] *= scale_x
            target_scaled[..., 1] *= scale_y
            target_scaled[..., 3] *= scale_y
        else:
            target_scaled = target_padded

        anchors_scaled = anchors * (target_size / 416.0)
        targets = self.build_target(
            anchors=anchors_scaled,
            img_size=(target_size, target_size),
            target_padded=target_scaled,
            maskGTboxes=mask_gt_boxes,
        )
        return imgs, targets, target_scaled

    def _compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        lamda_coord: float,
        lamda_noobj: float,
    ) -> torch.Tensor:
        """Compute YOLOv1-style MSE loss with sigmoid(xy/conf), exp(wh), and softmax(cls).

        Parameters:
        -----------
        preds : torch.Tensor
            Model prediction tensor of shape (B, 5*(5+num_classes), grid_h, grid_w).
        targets : torch.Tensor
            Target grid tensor of shape (B, 5*(5+num_classes), grid_h, grid_w).
        lamda_coord : float
            Weight multiplier for coordinate error loss component.
        lamda_noobj : float
            Weight multiplier for background / no-object confidence error loss component.

        Returns:
        --------
        torch.Tensor
            Scalar loss tensor representing total batch loss.
        """
        batch_size, _, grid_h, grid_w = preds.shape
        num_anchors = 5
        preds_u = (
            preds.float()
            .view(batch_size, num_anchors, 5 + self.num_classes, grid_h, grid_w)
            .permute(0, 3, 4, 1, 2)
        )
        targets_u = (
            targets.float()
            .view(batch_size, num_anchors, 5 + self.num_classes, grid_h, grid_w)
            .permute(0, 3, 4, 1, 2)
        )

        pred_xy = torch.sigmoid(preds_u[..., :2])
        pred_wh = torch.exp(preds_u[..., 2:4].clamp(-10.0, 10.0))
        pred_conf = torch.sigmoid(preds_u[..., 4])
        pred_cls = F.softmax(preds_u[..., 5:], dim=-1)

        obj_f = targets_u[..., 4]
        noobj_f = 1.0 - obj_f

        loss_bbx = (
            ((pred_xy - targets_u[..., :2]) ** 2).sum(dim=-1) * obj_f
            + ((pred_wh - targets_u[..., 2:4]) ** 2).sum(dim=-1) * obj_f
        ).sum() / batch_size
        conf_sq_err = (pred_conf - obj_f) ** 2
        loss_object = (conf_sq_err * obj_f).sum() / batch_size
        loss_noobj = (conf_sq_err * noobj_f).sum() / batch_size
        loss_class = (((pred_cls - targets_u[..., 5:]) ** 2).sum(dim=-1) * obj_f).sum() / batch_size

        return lamda_coord * loss_bbx + (loss_object + lamda_noobj * loss_noobj) + loss_class

    def _forward_loss(
        self,
        imgs: torch.Tensor,
        target_padded: torch.Tensor,
        mask_gt_boxes: torch.Tensor,
        anchors: torch.Tensor,
        img_size: int,
        device_obj: torch.device,
        pin_memory: bool,
        device_type: str,
        use_amp: bool,
        lamda_coord: float,
        lamda_noobj: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch, run forward pass under AMP, and compute loss.

        Parameters:
        -----------
        imgs : torch.Tensor
            Batch of input images of shape (B, 3, H, W).
        target_padded : torch.Tensor
            Padded targets tensor of shape (B, M, 5).
        mask_gt_boxes : torch.Tensor
            Boolean mask of shape (B, M).
        anchors : torch.Tensor
            Anchor box tensor of shape (5, 2).
        img_size : int
            Square image dimension to which batch is resized.
        device_obj : torch.device
            Compute device target.
        pin_memory : bool
            Whether non-blocking memory transfer is enabled.
        device_type : str
            Device type string ('cuda' or 'cpu').
        use_amp : bool
            Whether AMP is enabled.
        lamda_coord : float
            Weight multiplier for coordinate loss.
        lamda_noobj : float
            Weight multiplier for no-object confidence loss.

        Returns:
        --------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - loss: Scalar loss tensor.
            - preds: Raw model prediction logits of shape (B, 5*(5+num_classes), H//32, W//32).
            - target_scaled: Scaled ground truth box tensor of shape (B, M, 5).
        """
        imgs = imgs.to(device_obj, non_blocking=pin_memory)
        target_padded = target_padded.to(device_obj, non_blocking=pin_memory)
        mask_gt_boxes = mask_gt_boxes.to(device_obj, non_blocking=pin_memory)
        imgs_resized, targets, target_scaled = self._prepare_batch(
            imgs, target_padded, mask_gt_boxes, anchors, img_size
        )
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            preds = self(imgs_resized)
        loss = self._compute_loss(preds, targets, lamda_coord, lamda_noobj)
        return loss, preds, target_scaled

    def _train_epoch(
        self,
        train_loader: DataLoader,
        anchors: torch.Tensor,
        multi_scale_sizes: list[int],
        train_size: int,
        global_step: int,
        device_obj: torch.device,
        pin_memory: bool,
        device_type: str,
        use_amp: bool,
        lamda_coord: float,
        lamda_noobj: float,
        epoch: int,
        total_epochs: int,
        compute_mAP: bool = True,
        mAP_conf_threshold: float = 0.01,
        mAP_iou_threshold: float = 0.45,
    ) -> tuple[float, int, int, dict[str, float] | None]:
        """Run one training epoch in self.train() mode with AMP, changing size every 10 batches.

        Parameters:
        -----------
        train_loader : DataLoader
            Training DataLoader yielding batches of (imgs, target_padded, mask_gt_boxes).
        anchors : torch.Tensor
            Anchor box dimensions tensor of shape (5, 2) on device_obj.
        multi_scale_sizes : list[int]
            List of multi-scale square resolutions (e.g. [320, ..., 800]) sampled every 10 steps.
        train_size : int
            Current image size in pixels used for batch resizing.
        global_step : int
            Global iteration counter across all training epochs.
        device_obj : torch.device
            Target compute device (torch.device('cuda') or torch.device('cpu')).
        pin_memory : bool
            Whether non-blocking memory transfer is enabled when copying batches to device.
        device_type : str
            Device type string ('cuda' or 'cpu') for torch.amp.autocast and GradScaler.
        use_amp : bool
            Whether Automatic Mixed Precision (AMP) is enabled.
        lamda_coord : float
            Loss weighting coefficient for bounding box coordinate errors.
        lamda_noobj : float
            Loss weighting coefficient for background/no-object confidence errors.
        epoch : int
            Current epoch index (1-based) displayed in the progress bar.
        total_epochs : int
            Total number of epochs for display in the progress bar.
        compute_mAP : bool, default=True
            Whether to calculate detection accuracy metrics (mAP, mAP_50, mAP_75) on the training set.
        mAP_conf_threshold : float, default=0.01
            Confidence score threshold for NMS decoding during training mAP accumulation.
        mAP_iou_threshold : float, default=0.45
            IoU threshold for NMS suppression during training mAP accumulation.

        Returns:
        --------
        tuple[float, int, int, dict[str, float] | None]
            A tuple containing:
                - avg_train_loss: Float representing the average training loss over the epoch.
                - train_size: The most recent multi-scale resolution used in training.
                - global_step: Updated global training step counter.
                - train_mAP_metrics: Dict of {'mAP', 'mAP_50', 'mAP_75'} if compute_mAP=True, else None.
        """
        self.train()
        loss_sum = 0.0
        mAP_metric = None
        if compute_mAP:
            if MeanAveragePrecision is None:
                raise ImportError(
                    "torchmetrics is required for mAP evaluation. Install it via 'pip install torchmetrics>=1.0.0'."
                )
            mAP_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
            mAP_metric.warn_on_many_detections = False

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
        for step, (imgs, target_padded, mask_gt_boxes) in enumerate(pbar, start=1):
            if global_step > 0 and global_step % 10 == 0:
                idx = int(torch.randint(0, len(multi_scale_sizes), (1,)).item())
                train_size = multi_scale_sizes[idx]

            self.optimizer.zero_grad()
            loss, preds, target_scaled = self._forward_loss(
                imgs,
                target_padded,
                mask_gt_boxes,
                anchors,
                train_size,
                device_obj,
                pin_memory,
                device_type,
                use_amp,
                lamda_coord,
                lamda_noobj,
            )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if compute_mAP and mAP_metric is not None:
                with torch.no_grad():
                    preds_decoded = self.decode(
                        preds=preds.detach(),
                        anchors=anchors,
                        img_size=(train_size, train_size),
                        conf_threshold=mAP_conf_threshold,
                        iou_threshold=mAP_iou_threshold,
                    )
                    preds_metric = [
                        {
                            "boxes": d["boxes"].detach().cpu(),
                            "scores": d["scores"].detach().cpu(),
                            "labels": d["labels"].detach().cpu(),
                        }
                        for d in preds_decoded
                    ]
                    gt_targets = self._convert_gt_to_metric_format(target_scaled, mask_gt_boxes.to(device_obj), train_size)
                    mAP_metric.update(preds_metric, gt_targets)

            global_step += 1
            loss_sum += float(loss.item())
            pbar.set_postfix(
                size=f"{train_size}x{train_size}",
                loss=f"{loss.item():.4f}",
                avg_loss=f"{loss_sum / step:.4f}",
            )

        avg_loss = loss_sum / max(1, len(train_loader))
        train_mAP_metrics = None
        if compute_mAP and mAP_metric is not None:
            raw_metrics = mAP_metric.compute()
            train_mAP_metrics = {
                "mAP": float(raw_metrics["map"].item()),
                "mAP_50": float(raw_metrics["map_50"].item()),
                "mAP_75": float(raw_metrics["map_75"].item()),
            }

        return avg_loss, train_size, global_step, train_mAP_metrics

    def _eval_epoch(
        self,
        val_loader: DataLoader,
        anchors: torch.Tensor,
        val_size: int,
        device_obj: torch.device,
        pin_memory: bool,
        device_type: str,
        use_amp: bool,
        lamda_coord: float,
        lamda_noobj: float,
        epoch: int,
        total_epochs: int,
        compute_mAP: bool = True,
        mAP_conf_threshold: float = 0.01,
        mAP_iou_threshold: float = 0.45,
    ) -> tuple[float, dict[str, float] | None]:
        """Run one validation epoch in eval mode, computing validation loss and mAP metrics.

        Parameters:
        -----------
        val_loader : DataLoader
            Validation DataLoader yielding batches of (imgs, target_padded, mask_gt_boxes).
        anchors : torch.Tensor
            Anchor box tensor of shape (5, 2) on device_obj.
        val_size : int
            Image size (e.g. 416) for validation batch resizing.
        device_obj : torch.device
            Target compute device (torch.device('cuda') or torch.device('cpu')).
        pin_memory : bool
            Whether non-blocking memory transfer is enabled when copying batches to device.
        device_type : str
            Device type string ('cuda' or 'cpu') passed to torch.amp.autocast.
        use_amp : bool
            Whether Automatic Mixed Precision (AMP) is enabled during forward pass.
        lamda_coord : float
            Loss weighting coefficient for bounding box coordinate errors (bx, by, bw, bh).
        lamda_noobj : float
            Loss weighting coefficient for background/no-object confidence errors.
        epoch : int
            Current epoch index (1-based) displayed in the progress bar.
        total_epochs : int
            Total number of epochs for display in the progress bar.
        compute_mAP : bool, default=True
            Whether to calculate detection accuracy metrics (mAP, mAP_50, mAP_75) on the validation set.
        mAP_conf_threshold : float, default=0.01
            Confidence score threshold for NMS decoding during mAP evaluation.
        mAP_iou_threshold : float, default=0.45
            IoU threshold for NMS suppression during mAP evaluation.

        Returns:
        --------
        tuple[float, dict[str, float] | None]
            A tuple containing:
                - avg_val_loss: Float representing the average validation loss over the epoch.
                - val_mAP_metrics: Dict of {'mAP', 'mAP_50', 'mAP_75'} if compute_mAP=True, else None.
        """
        self.eval()
        loss_sum = 0.0
        mAP_metric = None
        if compute_mAP:
            if MeanAveragePrecision is None:
                raise ImportError(
                    "torchmetrics is required for mAP evaluation. Install it via 'pip install torchmetrics>=1.0.0'."
                )
            mAP_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
            mAP_metric.warn_on_many_detections = False

        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs} [Val {val_size}x{val_size}]")
        with torch.no_grad():
            for step, (imgs, target_padded, mask_gt_boxes) in enumerate(pbar, start=1):
                loss, preds, target_scaled = self._forward_loss(
                    imgs,
                    target_padded,
                    mask_gt_boxes,
                    anchors,
                    val_size,
                    device_obj,
                    pin_memory,
                    device_type,
                    use_amp,
                    lamda_coord,
                    lamda_noobj,
                )
                loss_sum += float(loss.item())

                if compute_mAP and mAP_metric is not None:
                    preds_decoded = self.decode(
                        preds=preds,
                        anchors=anchors,
                        img_size=(val_size, val_size),
                        conf_threshold=mAP_conf_threshold,
                        iou_threshold=mAP_iou_threshold,
                    )
                    preds_metric = [
                        {
                            "boxes": d["boxes"].detach().cpu(),
                            "scores": d["scores"].detach().cpu(),
                            "labels": d["labels"].detach().cpu(),
                        }
                        for d in preds_decoded
                    ]
                    gt_targets = self._convert_gt_to_metric_format(target_scaled, mask_gt_boxes, val_size)
                    mAP_metric.update(preds_metric, gt_targets)

                pbar.set_postfix(val_loss=f"{loss_sum / step:.4f}")

        avg_val_loss = loss_sum / max(1, len(val_loader))
        val_mAP_metrics = None
        if compute_mAP and mAP_metric is not None:
            raw_metrics = mAP_metric.compute()
            val_mAP_metrics = {
                "mAP": float(raw_metrics["map"].item()),
                "mAP_50": float(raw_metrics["map_50"].item()),
                "mAP_75": float(raw_metrics["map_75"].item()),
            }

        return avg_val_loss, val_mAP_metrics

    @torch.no_grad()
    def evaluate_mAP(
        self,
        dataloader: DataLoader,
        anchors: torch.Tensor | None = None,
        img_size: int = 416,
        mAP_conf_threshold: float = 0.01,
        mAP_iou_threshold: float = 0.45,
        device: str | torch.device | None = None,
        class_metrics: bool = False,
    ) -> dict[str, float | list[float]]:
        """Evaluate detection accuracy using MeanAveragePrecision over a DataLoader, skipping mAR.

        Parameters:
        -----------
        dataloader : DataLoader
            PyTorch DataLoader yielding (imgs, target_padded, mask_gt_boxes) batches
            to evaluate (e.g., train_loader or val_loader).
        anchors : torch.Tensor | None, default=None
            Anchor box dimensions tensor of shape (5, 2). If None, uses discovered anchors or
            default configuration anchors.
        img_size : int, default=416
            Square image dimension in pixels (e.g. 416) to which input images are resized for inference.
        mAP_conf_threshold : float, default=0.01
            Minimum object confidence threshold used during NMS decoding for mAP calculation.
            A low threshold (0.01) is standard to capture the full precision-recall curve.
        mAP_iou_threshold : float, default=0.45
            IoU threshold for Non-Maximum Suppression (NMS) during detection decoding.
        device : str | torch.device | None, default=None
            Device on which to execute evaluation (e.g., 'cuda' or 'cpu'). Defaults to current model device.
        class_metrics : bool, default=False
            Whether to calculate and return individual per-class mAP values in addition to overall mAP.

        Returns:
        --------
        dict[str, float | list[float]]
            Dictionary containing strictly mAP metrics (all mAR metrics skipped):
                - 'mAP': Mean Average Precision averaged over IoU thresholds [0.50:0.95].
                - 'mAP_50': Mean Average Precision at IoU threshold 0.50.
                - 'mAP_75': Mean Average Precision at IoU threshold 0.75.
                - 'mAP_per_class': (optional) List of per-class mAP values when class_metrics=True.
        """
        if MeanAveragePrecision is None:
            raise ImportError(
                "torchmetrics is required for evaluate_mAP. Install it via 'pip install torchmetrics>=1.0.0'."
            )

        if device is None:
            device = next(self.parameters()).device
        device_obj = torch.device(device)
        self.to(device_obj)
        self.eval()

        if anchors is None:
            if hasattr(self, "anchors") and self.anchors is not None:
                anchors_tensor = self.anchors
            elif hasattr(dataloader.dataset, "__len__") and len(dataloader.dataset) > 0:
                anchors_tensor = self.find_best_anchor_shape(dataloader.dataset)
            else:
                anchors_tensor = torch.tensor(
                    [
                        [33.9074, 43.1555],
                        [94.8374, 118.0962],
                        [159.8270, 256.6951],
                        [314.6092, 157.3282],
                        [375.2903, 333.2395],
                    ],
                    dtype=torch.float32,
                )
        else:
            anchors_tensor = torch.as_tensor(anchors, dtype=torch.float32)
        anchors_tensor = anchors_tensor.to(device=device_obj, dtype=torch.float32)

        mAP_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=class_metrics,
        )
        mAP_metric.warn_on_many_detections = False

        pbar = tqdm(dataloader, desc=f"Evaluating mAP [{img_size}x{img_size}]")
        for imgs, target_padded, mask_gt_boxes in pbar:
            imgs = imgs.to(device_obj)
            target_padded = target_padded.to(device_obj)
            mask_gt_boxes = mask_gt_boxes.to(device_obj)

            imgs_resized, _, target_scaled = self._prepare_batch(
                imgs, target_padded, mask_gt_boxes, anchors_tensor, img_size
            )
            with torch.amp.autocast(device_type=device_obj.type, enabled=(device_obj.type == "cuda")):
                preds = self(imgs_resized)

            preds_decoded = self.decode(
                preds=preds,
                anchors=anchors_tensor,
                img_size=(img_size, img_size),
                conf_threshold=mAP_conf_threshold,
                iou_threshold=mAP_iou_threshold,
            )
            preds_metric = [
                {
                    "boxes": d["boxes"].detach().cpu(),
                    "scores": d["scores"].detach().cpu(),
                    "labels": d["labels"].detach().cpu(),
                }
                for d in preds_decoded
            ]
            gt_targets = self._convert_gt_to_metric_format(target_scaled, mask_gt_boxes, img_size)
            mAP_metric.update(preds_metric, gt_targets)

        raw = mAP_metric.compute()
        res: dict[str, float | list[float]] = {
            "mAP": float(raw["map"].item()),
            "mAP_50": float(raw["map_50"].item()),
            "mAP_75": float(raw["map_75"].item()),
        }
        if class_metrics and "map_per_class" in raw:
            res["mAP_per_class"] = raw["map_per_class"].tolist()

        return res

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        N_epochs: int = 10,
        val_dataset: torch.utils.data.Dataset | None = None,
        lr: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lamda_coord: float = 5,
        lamda_noobj: float = 0.5,
        compute_train_mAP: bool = True,
        compute_val_mAP: bool = True,
        eval_mAP_interval: int = 1,
        mAP_conf_threshold: float = 0.01,
        mAP_iou_threshold: float = 0.45,
    ) -> dict[str, list[float]]:
        """Train YOLOv2 with AMP, multi-scale resizing, and automatic mAP evaluation on train/val sets.

        Parameters:
        -----------
        dataset : torch.utils.data.Dataset
            The training dataset (e.g. VOCdataset for train split).
        batch_size : int, default=32
            Number of samples per training and validation batch.
        N_epochs : int, default=10
            Total number of training epochs to run.
        val_dataset : torch.utils.data.Dataset | None, default=None
            Optional validation dataset (e.g. VOCdataset for val split). If provided, evaluation is run.
        lr : float, default=0.001
            Base learning rate for the Adam optimizer (backbone lr is scaled to lr / 10).
        device : str, default='cuda' if available else 'cpu'
            Execution hardware target ('cuda' or 'cpu').
        lamda_coord : float, default=5
            Weight multiplier for bounding box coordinate loss component.
        lamda_noobj : float, default=0.5
            Weight multiplier for confidence loss on grid cells containing no objects.
        compute_train_mAP : bool, default=True
            Whether to compute detection accuracy metrics (mAP, mAP_50, mAP_75) on the training set.
        compute_val_mAP : bool, default=True
            Whether to compute detection accuracy metrics (mAP, mAP_50, mAP_75) on the validation set.
        eval_mAP_interval : int, default=1
            Frequency (in epochs) at which mAP calculation is performed (e.g., 1 = every epoch).
        mAP_conf_threshold : float, default=0.01
            Object confidence threshold for decoding predictions during mAP calculation.
        mAP_iou_threshold : float, default=0.45
            IoU threshold for NMS decoding during mAP calculation.

        Returns:
        --------
        dict[str, list[float]]
            Training history dictionary containing per-epoch metric lists:
                - 'train_loss': list of floats
                - 'val_loss': list of floats (if val_dataset is provided)
                - 'train_mAP', 'train_mAP_50', 'train_mAP_75': list of floats (if compute_train_mAP=True)
                - 'val_mAP', 'val_mAP_50', 'val_mAP_75': list of floats (if compute_val_mAP=True)
        """
        device_obj = torch.device(device)
        device_type = device_obj.type
        use_amp = device_type == "cuda"
        self.to(device_obj)

        anchors = self.find_best_anchor_shape(dataset).to(device=device_obj, dtype=torch.float32)
        collate = partial(collate_fn)
        pin_memory = device_type == "cuda"
        trainLoader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
        valLoader = None
        if val_dataset is not None:
            valLoader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=pin_memory,
                collate_fn=collate,
            )

        head_modules = [
            self.batch_norm1,
            self.batch_norm2,
            self.batch_norm3,
            self.batch_norm4,
            self.layer_conv1,
            self.layer_conv2,
            self.layer_conv3,
            self.layer_conv4,
        ]
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.backbone.parameters(), "lr": lr / 10, "weight_decay": 0.0005},
                *({"params": module.parameters(), "lr": lr, "weight_decay": 0.0005} for module in head_modules),
            ]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, gamma=0.5, step_size=20)
        self.scaler = GradScaler(device=device_type, enabled=use_amp)

        multi_scale_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672]
        train_size = 320
        global_step = 0

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_mAP": [],
            "train_mAP_50": [],
            "train_mAP_75": [],
            "val_mAP": [],
            "val_mAP_50": [],
            "val_mAP_75": [],
        }

        for epoch in range(1, N_epochs + 1):
            run_train_mAP = compute_train_mAP and (epoch % eval_mAP_interval == 0)
            run_val_mAP = compute_val_mAP and (epoch % eval_mAP_interval == 0)

            avg_train_loss, train_size, global_step, train_metrics = self._train_epoch(
                trainLoader,
                anchors,
                multi_scale_sizes,
                train_size,
                global_step,
                device_obj,
                pin_memory,
                device_type,
                use_amp,
                lamda_coord,
                lamda_noobj,
                epoch,
                N_epochs,
                compute_mAP=run_train_mAP,
                mAP_conf_threshold=mAP_conf_threshold,
                mAP_iou_threshold=mAP_iou_threshold,
            )
            self.scheduler.step()

            history["train_loss"].append(avg_train_loss)
            if train_metrics is not None:
                history["train_mAP"].append(train_metrics["mAP"])
                history["train_mAP_50"].append(train_metrics["mAP_50"])
                history["train_mAP_75"].append(train_metrics["mAP_75"])

            log_parts = [f"Epoch {epoch}/{N_epochs}", f"train_loss: {avg_train_loss:.4f}"]
            if train_metrics is not None:
                log_parts.append(f"train_mAP_50: {train_metrics['mAP_50']:.4f}")

            if valLoader is not None:
                avg_val_loss, val_metrics = self._eval_epoch(
                    valLoader,
                    anchors,
                    416,
                    device_obj,
                    pin_memory,
                    device_type,
                    use_amp,
                    lamda_coord,
                    lamda_noobj,
                    epoch,
                    N_epochs,
                    compute_mAP=run_val_mAP,
                    mAP_conf_threshold=mAP_conf_threshold,
                    mAP_iou_threshold=mAP_iou_threshold,
                )
                history["val_loss"].append(avg_val_loss)
                log_parts.append(f"val_loss: {avg_val_loss:.4f}")
                if val_metrics is not None:
                    history["val_mAP"].append(val_metrics["mAP"])
                    history["val_mAP_50"].append(val_metrics["mAP_50"])
                    history["val_mAP_75"].append(val_metrics["mAP_75"])
                    log_parts.append(f"val_mAP_50: {val_metrics['mAP_50']:.4f}")

            print(" - ".join(log_parts))

        return history

    @torch.no_grad()
    def decode(
        self,
        preds: torch.Tensor,
        anchors: torch.Tensor,
        img_size: tuple[int, int] = (416, 416),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        orig_size: tuple[int, int] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Decode raw YOLOv2 predictions into filtered (x1, y1, x2, y2) boxes using class-aware NMS.

        Parameters:
        -----------
        preds : torch.Tensor
            Raw model prediction logits of shape (B, 5*(5+num_classes), H//32, W//32).
        anchors : torch.Tensor
            Anchor box dimensions tensor of shape (5, 2).
        img_size : tuple[int, int], default=(416, 416)
            Resolution (height, width) corresponding to the network prediction grid.
        conf_threshold : float, default=0.25
            Confidence score threshold to filter out low-confidence candidate detections.
        iou_threshold : float, default=0.45
            IoU overlap threshold used by batched Non-Maximum Suppression (NMS).
        orig_size : tuple[int, int] | None, default=None
            Optional original image dimension (height, width) to scale output boxes back to.

        Returns:
        --------
        list[dict[str, torch.Tensor]]
            List of length B where each element is a dict with keys:
                - 'boxes': FloatTensor of shape (N, 4) in [x1, y1, x2, y2] format.
                - 'scores': FloatTensor of shape (N,) containing objectness * class scores.
                - 'labels': LongTensor of shape (N,) containing integer category IDs.
        """
        if preds.ndim != 4:
            raise ValueError(f"Expected preds shape (B, C, H, W), got {tuple(preds.shape)}")

        batch_size, channels, grid_h, grid_w = preds.shape
        num_anchors = 5
        expected_channels = num_anchors * (5 + self.num_classes)
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels, got {channels}")

        img_h, img_w = int(img_size[0]), int(img_size[1])
        stride = 32
        anchors = torch.as_tensor(anchors, device=preds.device, dtype=torch.float32)
        if anchors.shape != (num_anchors, 2):
            raise ValueError(f"Expected anchors shape ({num_anchors}, 2), got {tuple(anchors.shape)}")

        preds_u = (
            preds.float()
            .view(batch_size, num_anchors, 5 + self.num_classes, grid_h, grid_w)
            .permute(0, 3, 4, 1, 2)
        )

        pred_xy = torch.sigmoid(preds_u[..., :2])
        pred_wh = torch.exp(preds_u[..., 2:4].clamp(-10.0, 10.0))
        pred_conf = torch.sigmoid(preds_u[..., 4])
        pred_cls = F.softmax(preds_u[..., 5:], dim=-1)

        rows = torch.arange(grid_h, device=preds.device, dtype=torch.float32).view(1, grid_h, 1, 1)
        cols = torch.arange(grid_w, device=preds.device, dtype=torch.float32).view(1, 1, grid_w, 1)
        anchor_w = (anchors[:, 0] * (img_w / 416.0)).view(1, 1, 1, num_anchors)
        anchor_h = (anchors[:, 1] * (img_h / 416.0)).view(1, 1, 1, num_anchors)

        cx = (cols + pred_xy[..., 0]) * stride
        cy = (rows + pred_xy[..., 1]) * stride
        bw = pred_wh[..., 0] * anchor_w
        bh = pred_wh[..., 1] * anchor_h

        x1 = (cx - 0.5 * bw).clamp(0.0, float(img_w))
        y1 = (cy - 0.5 * bh).clamp(0.0, float(img_h))
        x2 = (cx + 0.5 * bw).clamp(0.0, float(img_w))
        y2 = (cy + 0.5 * bh).clamp(0.0, float(img_h))

        if orig_size is not None:
            orig_h, orig_w = int(orig_size[0]), int(orig_size[1])
            scale_x = orig_w / float(img_w)
            scale_y = orig_h / float(img_h)
            x1 = (x1 * scale_x).clamp(0.0, float(orig_w))
            x2 = (x2 * scale_x).clamp(0.0, float(orig_w))
            y1 = (y1 * scale_y).clamp(0.0, float(orig_h))
            y2 = (y2 * scale_y).clamp(0.0, float(orig_h))

        boxes_all = torch.stack([x1, y1, x2, y2], dim=-1).reshape(batch_size, -1, 4)
        cls_scores, cls_ids = torch.max(pred_cls, dim=-1)
        scores_all = (pred_conf * cls_scores).reshape(batch_size, -1)
        labels_all = cls_ids.reshape(batch_size, -1)

        results: list[dict[str, torch.Tensor]] = []
        for b in range(batch_size):
            boxes_b = boxes_all[b]
            scores_b = scores_all[b]
            labels_b = labels_all[b]

            valid_mask = (
                (scores_b >= conf_threshold)
                & (boxes_b[:, 2] > boxes_b[:, 0])
                & (boxes_b[:, 3] > boxes_b[:, 1])
            )
            boxes_b = boxes_b[valid_mask]
            scores_b = scores_b[valid_mask]
            labels_b = labels_b[valid_mask]

            if boxes_b.numel() > 0:
                keep_idx = batched_nms(boxes_b, scores_b, labels_b, iou_threshold)
                boxes_b = boxes_b[keep_idx]
                scores_b = scores_b[keep_idx]
                labels_b = labels_b[keep_idx]

            results.append({
                "boxes": boxes_b,
                "scores": scores_b,
                "labels": labels_b,
            })

        return results
