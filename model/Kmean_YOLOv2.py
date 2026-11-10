import torch
import torch.nn as nn
from torchvision.ops import box_iou
class Kmean(nn.Module):
    def __init__(self, num_anchors: int = 5, max_iterators: int = 100)-> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.max_iterators = max_iterators
    def IOU(self, boxA: torch.Tensor, boxB: torch.Tensor)-> float:
        centroi = boxA.clone()
        centroi = centroi.repeat(1,2)
        
        mask = torch.tensor([
            [-1.0/2, -1.0/2, 1.0/2, 1.0/2]
        ])
        centroi = centroi*mask + 448.0
        BoxGT = boxB.clone()
        BoxGT = BoxGT.repeat(1,2)
        BoxGT = BoxGT*mask + 448.0

        iou = box_iou(centroi, BoxGT)
        return iou
    def forward(self, dataset: torch.Tensor)-> torch.Tensor:
        '''
        args:
            - dataset is a tensor have shape (N, 2) where 2 is (W, H) of GT boxes
        '''
        index = torch.randperm(len(dataset))[:self.num_anchors]
        centroi = dataset[index]

        for epoch in range(1, self.max_iterators+1):
            iou = self.IOU(centroi, dataset)
            distance = 1 - iou

            # 1. Gán mỗi box vào cụm có khoảng cách nhỏ nhất (IoU lớn nhất)
            if distance.shape[0] == self.num_anchors:
                nearest_clusters = torch.argmin(distance, dim=0)
            else:
                nearest_clusters = torch.argmin(distance, dim=1)

            # 2. Cập nhật lại tâm centroid bằng trung bình cộng các box trong cụm
            new_centroi = torch.zeros_like(centroi)
            for k in range(self.num_anchors):
                assigned_points = dataset[nearest_clusters == k]
                if len(assigned_points) > 0:
                    new_centroi[k] = assigned_points.mean(dim=0)
                else:
                    new_centroi[k] = centroi[k]

            # 3. Kiểm tra điều kiện hội tụ (nếu tâm không đổi thì dừng sớm)
            if torch.allclose(centroi, new_centroi, atol=1e-4):
                break

            centroi = new_centroi

        # Sắp xếp các anchor boxes theo diện tích tăng dần (w * h)
        areas = centroi[:, 0] * centroi[:, 1]
        centroi = centroi[torch.argsort(areas)]

        return centroi