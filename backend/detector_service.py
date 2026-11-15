from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.YOLODetector import YOLOv2
from backend.schemas import DetectionBox, DetectionResponse


class YOLODetectorService:
    """Thread-safe YOLOv2 inference wrapper backed by config.json and model_weight.pth."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or (ROOT_DIR / "config.json")
        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        model_cfg = self.config["model"]
        self.num_classes: int = int(model_cfg["num_classes"])
        self.default_img_size: int = int(model_cfg.get("img_size", 416))
        self.classes: list[str] = list(self.config["classes"])
        if len(self.classes) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} class names in config.json, found {len(self.classes)}"
            )

        requested_device = os.getenv("YOLO_DEVICE")
        if requested_device:
            self.device = torch.device(requested_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.anchors = torch.tensor(model_cfg["anchors"], dtype=torch.float32, device=self.device)
        self.mean = torch.tensor(model_cfg["mean"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(model_cfg["std"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        self.model = YOLOv2(num_classes=self.num_classes)
        weight_rel = Path(model_cfg["weight_path"])
        weight_path = weight_rel if weight_rel.is_absolute() else (ROOT_DIR / weight_rel)
        if not weight_path.exists():
            raise FileNotFoundError(f"Model weight file not found: {weight_path}")

        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._infer_lock = threading.Lock()

    def _preprocess(self, image: Image.Image, target_size: int) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if (tensor.shape[2], tensor.shape[3]) != (target_size, target_size):
            tensor = F.interpolate(
                tensor,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
        return (tensor - self.mean) / self.std

    @staticmethod
    def _normalize_img_size(img_size: int | None, default_size: int) -> int:
        size = int(img_size or default_size)
        size = max(128, min(size, 960))
        return max(32, (size // 32) * 32)

    def predict_image_bytes(
        self,
        image_bytes: bytes,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int | None = None,
    ) -> DetectionResponse:
        if not image_bytes:
            raise ValueError("Empty image payload")

        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb_img = ImageOps.exif_transpose(img).convert("RGB")
        orig_w, orig_h = rgb_img.size

        target_size = self._normalize_img_size(img_size, self.default_img_size)
        start_time = time.perf_counter()

        with self._infer_lock, torch.inference_mode():
            input_tensor = self._preprocess(rgb_img, target_size)
            preds = self.model(input_tensor)
            decoded = self.model.decode(
                preds=preds,
                anchors=self.anchors,
                img_size=(target_size, target_size),
                conf_threshold=float(conf_threshold),
                iou_threshold=float(iou_threshold),
                orig_size=(orig_h, orig_w),
            )[0]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        boxes = decoded["boxes"].detach().cpu().tolist()
        scores = decoded["scores"].detach().cpu().tolist()
        labels = decoded["labels"].detach().cpu().tolist()

        detections = [
            DetectionBox(
                x1=round(float(box[0]), 2),
                y1=round(float(box[1]), 2),
                x2=round(float(box[2]), 2),
                y2=round(float(box[3]), 2),
                score=round(float(score), 4),
                class_id=int(cls_id),
                class_name=self.classes[int(cls_id)] if 0 <= int(cls_id) < len(self.classes) else str(cls_id),
            )
            for box, score, cls_id in zip(boxes, scores, labels)
        ]

        return DetectionResponse(
            width=orig_w,
            height=orig_h,
            inference_ms=round(elapsed_ms, 2),
            detections=detections,
        )

