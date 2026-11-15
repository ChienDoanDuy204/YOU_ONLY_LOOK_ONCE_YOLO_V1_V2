from pathlib import Path
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
class VOCdataset(Dataset):
    def __init__(self, data_dir:str = None, image_set = 'train', year: str = '2012',list_transformation: list = [], class2idx: dict = None) -> None:
        self.list_transformation = [transforms.ToTensor()]
        if len(list_transformation):
            self.list_transformation.extend(list_transformation)
        self.voc = VOCDetection(
            root= data_dir,
            year= year,
            image_set= image_set,
            download= True,
            transform= transforms.Compose(self.list_transformation)
        )
        self.class2idx  = class2idx
    def __len__(self)->int:
        return  len(self.voc)
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        img, target = self.voc[index]
        boxes = []
        labels = []
        objects = target['annotation']['object']
        for obj in objects:
            if self.class2idx is not None and len(self.class2idx):
                labels.append(self.class2idx[obj['name']])
            else:
                labels.append(obj['name'])
            x1 = float(obj['bndbox']['xmin'])
            y1 = float(obj['bndbox']['ymin'])
            x2 = float(obj['bndbox']['xmax'])
            y2 = float(obj['bndbox']['ymax'])
            boxes.append([x1+(x2-x1)/2, y1+(y2-y1)/2, x2-x1, y2-y1])
        return img, {'boxes':torch.tensor(boxes, dtype= torch.float32), 'labels':torch.tensor(labels, dtype= torch.float32)}

# padding img, label, bbx follow the size img largest and the have most bbx
def collate_fn(Batch):
    imgs, targets = zip(*Batch)
    # lấy size lớn nhất của ảnh trong Batch
    H_img_max = max(img.shape[1] for img in imgs)
    W_img_max = max(img.shape[2] for img in imgs)
    # Lấy số lượng GT boxes lớn nhất có trong Batch để padding
    num_GTboxes_max = max(len(tgt['boxes']) for tgt in targets)

    # shape (B, M, 5)
    target_padded = torch.zeros((
        len(imgs),
        num_GTboxes_max,
        5
    ))
    mask_GTboxes = torch.zeros((len(imgs),num_GTboxes_max), dtype= torch.bool)
    for k, target in enumerate(targets):
        num_GTboxes = len(target['boxes'])
        target_padded[k, :num_GTboxes, :4] = target['boxes']
        target_padded[k, :num_GTboxes, 4] = target['labels']
        mask_GTboxes[k, :num_GTboxes] = True
    img_padded = torch.zeros(len(imgs), 3, H_img_max, W_img_max)
    for k, img in enumerate(imgs):
        img_padded[k, :, :img.shape[1], :img.shape[2]] = img
    return img_padded, target_padded, mask_GTboxes