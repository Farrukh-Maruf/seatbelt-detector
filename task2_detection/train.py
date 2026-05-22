from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
DATA_YAML = REPO / "data" / "yolo" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=0)
    ap.add_argument("--name", default="kamar_yolov8s")
    args = ap.parse_args()

    assert DATA_YAML.exists(), (
        f"{DATA_YAML} not found — run `python tools/prepare_data.py` first."
    )

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(REPO / "task2_detection" / "runs"),
        name=args.name,
        # augmentation tuned for this dataset
        fliplr=0.0,        # NO horizontal flip — see docstring
        flipud=0.0,
        mosaic=1.0,        # keep mosaic: helps with the small, sparse driver regions
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.5,   # heavy value jitter: day vs IR vs glare
        degrees=5.0,       # small rotation only (top-down geometry is fairly fixed)
        translate=0.1, scale=0.5,
        amp=True,          # mixed precision — important on 6 GB
        seed=42,
        verbose=True,
    )
    best = REPO / "task2_detection" / "runs" / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best}")


if __name__ == "__main__":
    main()
