from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from task4_pipeline.seatbelt_detector import SeatbeltDetector  # noqa: E402

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS",
                            str(REPO / "task2_detection/runs/kamar_yolov8s/weights/best.pt"))
CLASSIFIER_WEIGHTS = os.getenv("CLASSIFIER_WEIGHTS",
                                str(REPO / "task3_classification/weights/mobilenet_v3_small.pt"))
BELT_THRESHOLD = float(os.getenv("BELT_THRESHOLD", "0.5"))

app = FastAPI(title="Kamar (Seatbelt) Detector", version="1.0")
detector: SeatbeltDetector | None = None
_last_fps = 0.0


@app.on_event("startup")
def load_model():
    global detector
    try:
        clf = CLASSIFIER_WEIGHTS if Path(CLASSIFIER_WEIGHTS).exists() else None
        if not Path(YOLO_WEIGHTS).exists():
            print(f"[startup] YOLO weights not found at {YOLO_WEIGHTS}; model not loaded")
            return
        detector = SeatbeltDetector(YOLO_WEIGHTS, clf, belt_threshold=BELT_THRESHOLD)
        print(f"[startup] model loaded (classifier={'yes' if clf else 'no'})")
    except Exception as e:  # noqa: BLE001
        print(f"[startup] failed to load model: {e}")
        detector = None


@app.get("/health")
def health():
    return {
        "status": "ok" if detector is not None else "model_not_loaded",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": detector is not None,
        "fps": round(_last_fps, 1),
    }


@app.post("/predict/image")
async def predict_image(image: UploadFile = File(...)):
    global _last_fps
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = await image.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Judaham katta file (>10MB)")

    arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422,
                            detail="File formati notogri; JPEG/PNG kutilmoqda")

    t0 = time.perf_counter()
    fr = detector.process_main_driver(frame)
    _last_fps = 1.0 / max(1e-6, (time.perf_counter() - t0))

    if not fr.driver_detected:
        return JSONResponse(status_code=200, content={
            "driver_detected": False, "seatbelt_worn": None,
            "confidence": None, "bbox": None,
            "inference_ms": round(fr.inference_ms, 2)})

    d = fr.drivers[0]
    return {
        "driver_detected": True,
        "seatbelt_worn": bool(d.seatbelt_worn),
        "confidence": round(d.det_conf, 4),
        "bbox": list(d.bbox),
        "inference_ms": round(fr.inference_ms, 2),
    }
