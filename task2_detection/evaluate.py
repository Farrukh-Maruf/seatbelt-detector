from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
DATA_YAML = REPO / "data" / "yolo" / "data.yaml"
VAL_IMAGES = REPO / "data" / "yolo" / "val" / "images"
OUT = REPO / "task2_detection" / "eval_out"
CLASS_NAMES = ["kamar_bor", "kamar_yoq"]


def measure_fps(model, device, n=60):
    """Measure PURE inference FPS. Frames are pre-loaded into RAM so disk I/O does not
    pollute the timing (reading 4096x2480 JPEGs from disk dominates otherwise)."""
    paths = list(VAL_IMAGES.glob("*.jpg"))[:n]
    if not paths:
        return 0.0
    frames = [cv2.imread(str(p)) for p in paths]
    model.predict(frames[0], device=device, verbose=False)  # warmup
    t0 = time.perf_counter()
    for f in frames:
        model.predict(f, device=device, verbose=False)
    dt = time.perf_counter() - t0
    return len(frames) / dt if dt else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    metrics = model.val(data=str(DATA_YAML), device=args.device, plots=True, verbose=False)

    print("\n=== Task 2 — Detection metrics ===")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}   (target >= 0.80)")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}   (target >= 0.55)")
    for i, name in enumerate(CLASS_NAMES):
        p = metrics.box.p[i] if i < len(metrics.box.p) else float("nan")
        r = metrics.box.r[i] if i < len(metrics.box.r) else float("nan")
        note = "  (target P>=0.85, R>=0.80)" if name == "kamar_yoq" else ""
        print(f"{name:10s} P={p:.4f}  R={r:.4f}{note}")

    fps = measure_fps(model, args.device)
    print(f"FPS          : {fps:.1f}   (target >= 30)")

    # confusion matrix is saved by val(plots=True) into the run dir; point to it
    print(f"\nConfusion matrix + PR curves saved under: {metrics.save_dir}")

    # dump val frames that have any detection disagreeing with GT count, for FP/FN review
    dumped = 0
    for img_path in list(VAL_IMAGES.glob("*.jpg")):
        res = model.predict(str(img_path), device=args.device, verbose=False)[0]
        lbl = VAL_IMAGES.parent / "labels" / f"{img_path.stem}.txt"
        gt_n = len(lbl.read_text().splitlines()) if lbl.exists() else 0
        if len(res.boxes) != gt_n:
            cv2.imwrite(str(OUT / f"mismatch_{img_path.stem}.jpg"), res.plot())
            dumped += 1
    print(f"Dumped {dumped} val frames with box-count mismatch -> {OUT}")


if __name__ == "__main__":
    main()
