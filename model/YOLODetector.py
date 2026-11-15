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

class DarkNet(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.extractor  = timm.create_model('resnet18.a3_in1k', pretrained=True)
        # Bỏ lớp AVG_POOLING và FC của ResNet
        self.extractor.global_pool = nn.Identity()
        self.extractor.fc = nn.Identity()
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = self.extractor(x)
        return x

class YOLOv2(nn.Module):
    def __init__(self, num_classes:int = 20)-> None:
        super().__init__()
        self.backbone = DarkNet()
        self.num_classes = num_classes
        self.batch_norm1 = nn.BatchNorm2d(
            num_features=512,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True
        )
        self.layer_conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride = 1, 
            padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(
            num_features=1024,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True
        )
        self.layer_conv2 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.batch_norm3 = nn.BatchNorm2d(
            num_features=1024,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True
        )
        self.layer_conv3 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.batch_norm4 = nn.BatchNorm2d(
            num_features=1024,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True
        )
        self.layer_conv4 = nn.Conv2d(
            in_channels=1024,
            out_channels=5*(5+num_classes),
            kernel_size=1,
            stride=1,
            padding=0
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:

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

    # find popolar size of bbx in GT boxes with Kmean clustering 
    def find_best_anchor_shape(self, dataset: torch.utils.data.Dataset, num_anchors: int = 5, max_iterators: int = 100)-> torch.Tensor:
        data_cluster = dataset[0][1]['boxes'].clone()[:,2:]
        for i in range(1, dataset.__len__()):
            data_cluster = torch.cat([data_cluster, dataset[i][1]['boxes'].clone()[:,2:]], dim=0)
        Kmean_cluster = Kmean(num_anchors=num_anchors, max_iterators=max_iterators)
        anchor = Kmean_cluster(data_cluster)
        return anchor
    
    def build_target(self,
        anchors: torch.Tensor,
        img_size:tuple[int, int] = (416, 416),
        target_padded: torch.Tensor= None, 
        maskGTboxes: torch.Tensor = None)->torch.Tensor:
        """Build YOLOv2 target tensor of shape (B, 5*(5+num_classes), H//32, W//32)."""
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

    def _prepare_batch(
        self,
        imgs: torch.Tensor,
        target_padded: torch.Tensor,
        mask_gt_boxes: torch.Tensor,
        anchors: torch.Tensor,
        target_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize image batch to (target_size, target_size), scale boxes, and build target tensor."""
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
        return imgs, targets

    def _compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        lamda_coord: float,
        lamda_noobj: float,
    ) -> torch.Tensor:
        """Compute YOLOv1-style MSE loss with sigmoid(xy/conf), exp(wh), and softmax(cls)."""
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
    ) -> torch.Tensor:
        """Prepare a batch, run AMP forward pass, and compute float32 YOLOv1 loss."""
        imgs = imgs.to(device_obj, non_blocking=pin_memory)
        target_padded = target_padded.to(device_obj, non_blocking=pin_memory)
        mask_gt_boxes = mask_gt_boxes.to(device_obj, non_blocking=pin_memory)
        imgs_resized, targets = self._prepare_batch(imgs, target_padded, mask_gt_boxes, anchors, img_size)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            preds = self(imgs_resized)
        return self._compute_loss(preds, targets, lamda_coord, lamda_noobj)

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
    ) -> tuple[float, int, int]:
        """Run one training epoch in self.train() mode with AMP, changing size every 10 batches."""
        self.train()
        loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
        for step, (imgs, target_padded, mask_gt_boxes) in enumerate(pbar, start=1):
            if global_step > 0 and global_step % 8 == 0:
                idx = int(torch.randint(0, len(multi_scale_sizes), (1,)).item())
                train_size = multi_scale_sizes[idx]

            self.optimizer.zero_grad()
            loss = self._forward_loss(
                imgs, target_padded, mask_gt_boxes, anchors, train_size,
                device_obj, pin_memory, device_type, use_amp, lamda_coord, lamda_noobj,
            )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            global_step += 1
            loss_sum += float(loss.item())
            pbar.set_postfix(
                size=f"{train_size}x{train_size}",
                loss=f"{loss.item():.4f}",
                avg_loss=f"{loss_sum / step:.4f}",
            )
        return loss_sum / max(1, len(train_loader)), train_size, global_step

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
    ) -> float:
        """Run one validation epoch in self.eval() mode with torch.no_grad()."""
        self.eval()
        loss_sum = 0.0
        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs} [Val {val_size}x{val_size}]")
        with torch.no_grad():
            for step, (imgs, target_padded, mask_gt_boxes) in enumerate(pbar, start=1):
                loss = self._forward_loss(
                    imgs, target_padded, mask_gt_boxes, anchors, val_size,
                    device_obj, pin_memory, device_type, use_amp, lamda_coord, lamda_noobj,
                )
                loss_sum += float(loss.item())
                pbar.set_postfix(val_loss=f"{loss_sum / step:.4f}")
        return loss_sum / max(1, len(val_loader))
    
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
    )->None:
        """Train YOLOv2 with AMP, multi-scale resizing every 10 batches, and optional validation."""
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
            self.batch_norm1, self.batch_norm2, self.batch_norm3, self.batch_norm4,
            self.layer_conv1, self.layer_conv2, self.layer_conv3, self.layer_conv4,
        ]
        self.optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters(), 'lr': lr / 10, 'weight_decay': 0.0005},
            *({'params': module.parameters(), 'lr': lr, 'weight_decay': 0.0005} for module in head_modules),
        ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, gamma=0.5, step_size=20)
        self.scaler = GradScaler(device=device_type, enabled=use_amp)

        multi_scale_sizes = [416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        train_size = 416
        global_step = 0

        for epoch in range(1, N_epochs + 1):
            avg_train_loss, train_size, global_step = self._train_epoch(
                trainLoader, anchors, multi_scale_sizes, train_size, global_step,
                device_obj, pin_memory, device_type, use_amp,
                lamda_coord, lamda_noobj, epoch, N_epochs,
            )
            self.scheduler.step()

            if valLoader is not None:
                avg_val_loss = self._eval_epoch(
                    valLoader, anchors, 416, device_obj, pin_memory, device_type, use_amp,
                    lamda_coord, lamda_noobj, epoch, N_epochs,
                )
                print(f"Epoch {epoch}/{N_epochs} - train_loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch}/{N_epochs} - train_loss: {avg_train_loss:.4f}")

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
        """Decode raw YOLOv2 predictions into filtered (x1, y1, x2, y2) boxes using class-aware NMS."""
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
