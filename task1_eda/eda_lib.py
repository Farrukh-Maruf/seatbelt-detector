from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]        
SRC_IMAGES = PROJECT / "rasmlar" / "images"
SRC_LABELS = PROJECT / "rasmlar" / "labels"
CLASS_NAMES = ["kamar_bor", "kamar_yoq"]

CITIES = ["Samarqand", "Andijon", "Toshkent"]


def _read_boxes(stem: str):
    p = SRC_LABELS / f"{stem}.txt"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            c, xc, yc, w, h = parts
            out.append((int(c), float(xc), float(yc), float(w), float(h)))
    return out


def is_grayscale(img: np.ndarray, tol: float = 8.0) -> bool:
    """True if the frame is effectively grayscale/IR (night)."""
    if img.ndim == 2:
        return True
    b, g, r = cv2.split(img.astype(np.float32))
    return float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))) < tol


def build_box_table() -> pd.DataFrame:
    """One row per labelled box: stem, class, normalised w/h/area, frame day/night,
    frame brightness. Reads each image once."""
    rows = []
    for img_path in sorted(SRC_IMAGES.glob("*.jpg")):
        stem = img_path.stem
        boxes = _read_boxes(stem)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = is_grayscale(img)
        bright = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
        if not boxes:
            rows.append(dict(stem=stem, cls=-1, cls_name="(empty)", w=np.nan,
                                h=np.nan, area=np.nan, night=gray, brightness=bright))
            continue
        for c, xc, yc, w, h in boxes:
            rows.append(dict(stem=stem, cls=c, cls_name=CLASS_NAMES[c], w=w, h=h,
                             area=w * h, night=gray, brightness=bright))
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    boxes = df[df.cls >= 0]
    c0 = int((boxes.cls == 0).sum())
    c1 = int((boxes.cls == 1).sum())
    total = c0 + c1
    per_frame = boxes.groupby("stem").size()
    return {
        "n_frames": int(df.stem.nunique()),
        "n_empty_frames": int((df.cls == -1).sum()),
        "n_boxes": total,
        "kamar_bor": c0,
        "kamar_yoq": c1,
        "imbalance_ratio": round(c1 / c0, 2) if c0 else float("inf"),
        "boxes_per_frame_mean": round(float(per_frame.mean()), 2),
        "boxes_per_frame_max": int(per_frame.max()),
        "pct_night_frames": round(100 * df.groupby("stem").night.first().mean(), 1),
    }
