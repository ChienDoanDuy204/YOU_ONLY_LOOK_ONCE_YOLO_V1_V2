from pathlib import Path
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch

VOC_CLASSES: list[str] = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS2INDEX: dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}


class VOCdataset(Dataset):
    """Pascal VOC Dataset wrapper with optional transforms and class-to-index mapping.

    Parameters:
    -----------
    data_dir : str | None, default=None
        Root directory where VOCdevkit is stored (e.g. data/train or data/val).
    image_set : str, default='train'
        Dataset split to load ('train', 'val', or 'trainval').
    year : str, default='2012'
        Dataset release year ('2007' or '2012').
    list_transformation : list, default=[]
        List of additional torchvision transforms to apply after ToTensor() (e.g. Normalize).
    class2idx : dict | None, default=None
        Mapping dictionary from class name string to integer index [0..19].
        Defaults to standard VOC CLASS2INDEX if None.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        image_set: str = "train",
        year: str = "2012",
        list_transformation: list | None = None,
        class2idx: dict[str, int] | None = None,
    ) -> None:
        self.list_transformation = [transforms.ToTensor()]
        if len(list_transformation):
            self.list_transformation.extend(list_transformation)
        self.voc = VOCDetection(
            root=data_dir,
            year=year,
            image_set=image_set,
            download=True,
            transform=transforms.Compose(self.list_transformation),
        )
        self.class2idx = CLASS2INDEX if class2idx is None else class2idx

    def __len__(self) -> int:
        """Return the total number of images in this dataset split."""
        return len(self.voc)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Retrieve image tensor and bounding box target dictionary at given index.

        Parameters:
        -----------
        index : int
            Dataset index of the sample to retrieve.

        Returns:
        --------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            - img: Image tensor of shape (3, H, W)
            - target: Dict containing:
                - 'boxes': FloatTensor of shape (num_objects, 4) in [cx, cy, w, h] format.
                - 'labels': FloatTensor of shape (num_objects,) containing class indices.
        """
        img, target = self.voc[index]
        boxes = []
        labels = []
        objects = target["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        for obj in objects:
            if self.class2idx is not None and obj["name"] in self.class2idx:
                labels.append(self.class2idx[obj["name"]])
            else:
                labels.append(obj["name"])
            x1 = float(obj["bndbox"]["xmin"])
            y1 = float(obj["bndbox"]["ymin"])
            x2 = float(obj["bndbox"]["xmax"])
            y2 = float(obj["bndbox"]["ymax"])
            boxes.append([x1 + (x2 - x1) / 2.0, y1 + (y2 - y1) / 2.0, x2 - x1, y2 - y1])

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.float32)

        return img, {"boxes": boxes_tensor, "labels": labels_tensor}


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-sized images and bounding boxes in a batch to uniform dimensions.

    Parameters:
    -----------
    batch : list[tuple[torch.Tensor, dict[str, torch.Tensor]]]
        List of (img, target) tuples returned by VOCdataset.__getitem__.

    Returns:
    --------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - img_padded: Tensor of shape (B, 3, H_max, W_max) with images placed at top-left.
        - target_padded: Tensor of shape (B, M_max, 5) with [cx, cy, w, h, label].
        - mask_gt_boxes: Boolean tensor of shape (B, M_max) indicating valid ground truth boxes.
    """
    imgs, targets = zip(*batch)
    h_img_max = max(img.shape[1] for img in imgs)
    w_img_max = max(img.shape[2] for img in imgs)
    num_gt_boxes_max = max(len(tgt["boxes"]) for tgt in targets)
    num_gt_boxes_max = max(1, num_gt_boxes_max)

    target_padded = torch.zeros((len(imgs), num_gt_boxes_max, 5), dtype=torch.float32)
    mask_gt_boxes = torch.zeros((len(imgs), num_gt_boxes_max), dtype=torch.bool)

    for k, target in enumerate(targets):
        num_gt = len(target["boxes"])
        if num_gt > 0:
            target_padded[k, :num_gt, :4] = target["boxes"]
            target_padded[k, :num_gt, 4] = target["labels"]
            mask_gt_boxes[k, :num_gt] = True

    img_padded = torch.zeros((len(imgs), 3, h_img_max, w_img_max), dtype=imgs[0].dtype)
    for k, img in enumerate(imgs):
        img_padded[k, :, :img.shape[1], :img.shape[2]] = img

    return img_padded, target_padded, mask_gt_boxes
