import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from model.Kmean_YOLOv2 import *
class DarkNet(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.extractor  = timm.create_model('resnet18.a3_in1k', pretrained=True)

        # Thay đổi stride của các lớp conv trong layer4 để tránh việc bị downsampling
        for name, module in self.extractor.layer4[0].named_modules():
            if isinstance(module, nn.Conv2d):
                module.stride = (1, 1)
        # Bỏ lớp AVG_POOLING và FC của ResNet
        self.extractor.global_pool = nn.Identity()
        self.extractor.fc = nn.Identity()
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = self.extractor(x)
        return x

class YOLOv2(nn.Module):
    def __init__(self, num_classes:int = 20)-> None:
        self.backbone = DarkNet()
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
        feature_maps = F.dropout(feature_maps, p = 0.5)
        feature_maps = F.relu(feature_maps, inplace=True)
        

        ## Convolution layer 1
        feature_maps = self.layer_conv1(feature_maps)
        feature_maps = self.batch_norm2(feature_maps)
        feature_maps = F.dropout(feature_maps, p = 0.5)
        feature_maps = F.relu(feature_maps, inplace=True)

        ## Convolution layer 2
        feature_maps = self.layer_conv2(feature_maps)
        feature_maps = self.batch_norm3(feature_maps)
        feature_maps = F.dropout(feature_maps, p = 0.5)
        feature_maps = F.relu(feature_maps, inplace=True)

        ## Convoltuion layer 3
        feature_maps = self.layer_conv3(feature_maps)
        feature_maps = self.batch_norm4(feature_maps)
        feature_maps = F.dropout(feature_maps, p = 0.5)
        feature_maps = F.relu(feature_maps, inplace=True)


        ## Head prediction 
        output = self.layer_conv4(feature_maps)
        return output
    
    def find_best_anchor_shape(self, dataset: torch.utils.data.Dataset, num_anchors: int = 5, max_iterators: int = 100)-> torch.Tensor:
        data_cluster = dataset[0][1]['boxes'].clone()[:,2:]
        for i in range(1, dataset.__len__()):
            data_cluster = torch.cat([data_cluster, dataset[i][1]['boxes'].clone()[:,2:]], dim=0)
        Kmean_cluster = Kmean(num_anchors=num_anchors, max_iterators=max_iterators)
        anchor = Kmean_cluster(data_cluster)
        return anchor
    def build_target(self, anchors: torch.Tensor, img_size:tuple[int, int] = (416, 416)):
        pass
