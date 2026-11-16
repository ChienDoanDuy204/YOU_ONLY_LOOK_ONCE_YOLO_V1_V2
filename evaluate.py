#!/usr/bin/env python3
"""Standalone CLI script to evaluate YOLOv2 detection accuracy (mAP, mAP_50, mAP_75) on VOC splits.

All mAR metrics are skipped. All outputs strictly follow 'mAP' capitalization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.YOLODetector import YOLOv2
from src.data.orgainize_data import VOCdataset, collate_fn, VOC_CLASSES, CLASS2INDEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv2 model mAP on Pascal VOC 2012 train and/or val splits."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "both"],
        help="Dataset split to evaluate ('train', 'val', or 'both'). Default: val",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="model_weight.pth",
        help="Path to trained model weights (.pth). Default: model_weight.pth",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config.json. Default: config.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation DataLoader. Default: 16",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=416,
        help="Evaluation image dimension in pixels. Default: 416",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.01,
        help="Confidence threshold for decoding detections during mAP evaluation. Default: 0.01",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS during mAP evaluation. Default: 0.45",
    )
    parser.add_argument(
        "--class-metrics",
        action="store_true",
        help="Calculate and display per-class mAP breakdown.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda' or 'cpu'). Defaults to CUDA if available.",
    )
    return parser.parse_args()


def print_mAP_report(
    split_name: str,
    metrics: dict[str, float | list[float]],
    classes: list[str],
    num_samples: int,
) -> None:
    """Print a clean formatted mAP summary table to stdout."""
    print("\n" + "=" * 60)
    print(f" YOLOv2 mAP EVALUATION REPORT: {split_name.upper()} SET ({num_samples} images)")
    print("=" * 60)
    print(f"  • mAP @ [0.50:0.95] : {metrics['mAP']:.4f} ({metrics['mAP'] * 100:.2f}%)")
    print(f"  • mAP @ 0.50 (VOC)  : {metrics['mAP_50']:.4f} ({metrics['mAP_50'] * 100:.2f}%)")
    print(f"  • mAP @ 0.75        : {metrics['mAP_75']:.4f} ({metrics['mAP_75'] * 100:.2f}%)")
    print("=" * 60)

    if "mAP_per_class" in metrics and isinstance(metrics["mAP_per_class"], list):
        print("\nPer-Class mAP @ [0.50:0.95] Breakdown:")
        print(f"  {'Index':<6} {'Class Name':<16} {'mAP':<10}")
        print("  " + "-" * 34)
        for idx, (cls_name, ap) in enumerate(zip(classes, metrics["mAP_per_class"])):
            print(f"  {idx:<6} {cls_name:<16} {ap:.4f}")
        print("  " + "-" * 34)
    print()


def run_evaluation() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_cfg = config.get("model", {})
    classes = config.get("classes", VOC_CLASSES)
    num_classes = int(model_cfg.get("num_classes", len(classes)))
    anchors = torch.tensor(model_cfg.get("anchors", []), dtype=torch.float32)
    mean = model_cfg.get("mean", [0.485, 0.456, 0.406])
    std = model_cfg.get("std", [0.229, 0.224, 0.225])

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Build model and load weights
    model = YOLOv2(num_classes=num_classes)
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = ROOT_DIR / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print(f"Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    splits_to_run = ["train", "val"] if args.split == "both" else [args.split]

    for split in splits_to_run:
        split_dir = ROOT_DIR / "data" / split
        print(f"\nInitializing {split} dataset from {split_dir}...")
        dataset = VOCdataset(
            data_dir=str(split_dir),
            image_set=split,
            year="2012",
            list_transformation=[
                transforms.Normalize(mean=mean, std=std),
            ],
            class2idx=CLASS2INDEX,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
            collate_fn=partial(collate_fn),
        )

        metrics = model.evaluate_mAP(
            dataloader=dataloader,
            anchors=anchors if anchors.numel() > 0 else None,
            img_size=args.img_size,
            mAP_conf_threshold=args.conf_threshold,
            mAP_iou_threshold=args.iou_threshold,
            device=device,
            class_metrics=args.class_metrics,
        )

        print_mAP_report(split, metrics, classes, len(dataset))


if __name__ == "__main__":
    run_evaluation()
