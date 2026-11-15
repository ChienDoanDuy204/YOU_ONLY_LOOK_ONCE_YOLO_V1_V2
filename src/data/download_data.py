from pathlib import Path
from torchvision.datasets import VOCDetection
import os

if __name__ == '__main__':
    root = Path(__file__).parent.parent.parent
    data_dir = str(root / 'data')
    train_dir = str(root/'data'/'train')
    val_dir = str(root/'data'/'val')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    voc_train = VOCDetection(root=train_dir, year="2012", image_set="train", download=True)
    voc_val = VOCDetection(root=val_dir, year="2012", image_set="val", download=True)