from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from task3_classification.smoothing import TemporalSmoother  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DriverResult:
    bbox: tuple[int, int, int, int]          # x1,y1,x2,y2  frame pixel da
    seatbelt_worn: bool                       # smoothed final decision
    p_belt: float                             # classifier P(kamar_bor)
    det_conf: float                           # YOLO detection confidence


@dataclass
class FrameResult:
    driver_detected: bool
    drivers: list[DriverResult] = field(default_factory=list)
    inference_ms: float = 0.0


def _build_classifier(arch: str) -> nn.Module:
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    else:
        raise ValueError(arch)
    return m


class SeatbeltDetector:
    def __init__(self, yolo_weights: str, classifier_weights: str | None = None,
                    belt_threshold: float = 0.5, det_conf: float = 0.35,
                    clahe: bool = False, device: str | None = None,
                    smoother_maxlen: int = 10):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(yolo_weights)
        self.belt_threshold = belt_threshold
        self.det_conf = det_conf
        self.clahe = clahe
        self.smoother_maxlen = smoother_maxlen
        self._smoothers: dict[int, TemporalSmoother] = {}

        self.classifier = None
        if classifier_weights:
            ckpt = torch.load(classifier_weights, map_location="cpu")
            self.classifier = _build_classifier(ckpt.get("arch", "mobilenet_v3_small"))
            self.classifier.load_state_dict(ckpt["state_dict"])
            self.classifier.to(self.device).eval()
            self._bor_idx = ckpt["class_to_idx"]["kamar_bor"]
        self._tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    # ---- preprocessing -------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if not self.clahe:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _classify_crop(self, crop: np.ndarray) -> float:
        """Return P(kamar_bor) for a BGR crop. If no classifier is loaded, fall back to
        the YOLO class (handled by caller)."""
        rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
        x = self._tf(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(torch.sigmoid(self.classifier(x)).item())

    # ---- main API ------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        proc = self._preprocess(frame)
        res = self.yolo.predict(proc, conf=self.det_conf, device=self.device,
                                verbose=False)[0]

        drivers: list[DriverResult] = []
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            det_conf = float(box.conf[0])
            yolo_cls = int(box.cls[0])              # 0=kamar_bor, 1=kamar_yoq
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            if self.classifier is not None:
                p_belt = self._classify_crop(crop)
            else:
                # no Stage-2 model: trust YOLO's belt class
                p_belt = 1.0 if yolo_cls == 0 else 0.0

            raw_violation = p_belt < self.belt_threshold      # kamar_yoq if low P(belt)
            sm = self._smoothers.setdefault(
                i, TemporalSmoother(self.smoother_maxlen))
            stable_violation = sm.update(raw_violation)
            drivers.append(DriverResult(
                bbox=(x1, y1, x2, y2),
                seatbelt_worn=not stable_violation,
                p_belt=p_belt, det_conf=det_conf))

        dt = (time.perf_counter() - t0) * 1000
        return FrameResult(driver_detected=len(drivers) > 0,
                            drivers=drivers, inference_ms=dt)

    def process_main_driver(self, frame: np.ndarray) -> FrameResult:
        """Single-answer variant for the API: keep only the dominant (largest-area)
        driver. Handles the 'two people in frame' case by choosing the main lane car."""
        fr = self.process_frame(frame)
        if not fr.drivers:
            return fr
        main = max(fr.drivers, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        fr.drivers = [main]
        return fr

    def annotate(self, frame: np.ndarray, fr: FrameResult, fps: float | None = None):
        out = frame.copy()
        for d in fr.drivers:
            x1, y1, x2, y2 = d.bbox
            worn = d.seatbelt_worn
            color = (0, 180, 0) if worn else (0, 0, 230)
            label = f"{'kamar_bor' if worn else 'KAMAR_YOQ'} p={d.p_belt:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if fps is not None:
            cv2.putText(out, f"FPS {fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return out

    def process_video(self, source, save_path: str | None = None, show: bool = False):
        """Accept a video file path or RTSP URL. Writes annotated video and dumps
        frames where a violation was committed (for FP/FN review)."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source: {source}")
        writer = None
        self._smoothers.clear()
        n, t0 = 0, time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fr = self.process_frame(frame)
            n += 1
            fps = n / (time.perf_counter() - t0)
            vis = self.annotate(frame, fr, fps)
            if save_path:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(save_path,
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             25, (w, h))
                writer.write(vis)
            if show:
                cv2.imshow("seatbelt", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Processed {n} frames, avg FPS {n / (time.perf_counter() - t0):.1f}")
