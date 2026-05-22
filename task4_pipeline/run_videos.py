from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import torch

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
sys.path.insert(0, str(REPO))
from task4_pipeline.seatbelt_detector import SeatbeltDetector  # noqa: E402

YOLO_W = REPO / "task2_detection/runs/kamar_yolov8s/weights/best.pt"
CLF_W = REPO / "task3_classification/weights/mobilenet_v3_small.pt"
# CLF_W = REPO / "task3_classification/weights/efficientnet_b0.pt"
VIDEO_DIR = PROJECT / "test_videolar"
OUT_DIR = REPO / "task4_pipeline" / "out"
BELT_THRESHOLD = 0.586   # Youden-J from Task 3 (production)


def run_one(det: SeatbeltDetector, path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  [skip] cannot open {path.name}")
        return None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{path.stem}_annotated_efficientnet.mp4"
    writer = None
    det._smoothers.clear()

    n = 0
    frames_with_driver = 0
    violations = 0
    latencies = []
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr = det.process_frame(frame)
        n += 1
        latencies.append(fr.inference_ms)
        if fr.driver_detected:
            frames_with_driver += 1
            violations += sum(1 for d in fr.drivers if not d.seatbelt_worn)
        live_fps = n / (time.perf_counter() - t0)
        vis = det.annotate(frame, fr, live_fps)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(out_path),
                                        cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        writer.write(vis)
    cap.release()
    if writer:
        writer.release()

    wall = time.perf_counter() - t0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    gpu_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0
    return {
        "video": path.name, "frames": n, "fps": n / wall,
        "avg_latency_ms": avg_lat,
        "frames_with_driver": frames_with_driver,
        "violation_detections": violations,
        "gpu_mb_peak": gpu_mb, "out": out_path.name,
    }


def main():
    assert YOLO_W.exists(), f"YOLO weights missing: {YOLO_W}"
    clf = str(CLF_W) if CLF_W.exists() else None
    det = SeatbeltDetector(str(YOLO_W), clf, belt_threshold=BELT_THRESHOLD)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"Classifier loaded: {'yes' if clf else 'no (YOLO class fallback)'}")
    print(f"Belt threshold (Youden-J): {BELT_THRESHOLD}\n")

    results = []
    for v in sorted(VIDEO_DIR.glob("*.mp4")):
        print(f"Processing {v.name} ...")
        r = run_one(det, v)
        if r:
            results.append(r)
            print(f"  frames={r['frames']}  FPS={r['fps']:.1f}  "
                    f"latency={r['avg_latency_ms']:.1f}ms  "
                    f"driver_frames={r['frames_with_driver']}  "
                    f"violations={r['violation_detections']}  "
                    f"GPU_peak={r['gpu_mb_peak']:.0f}MB  -> {r['out']}")

    if results:
        avg_fps = sum(r["fps"] for r in results) / len(results)
        avg_lat = sum(r["avg_latency_ms"] for r in results) / len(results)
        peak = max(r["gpu_mb_peak"] for r in results)
        print("\n=== SUMMARY (Task 4 targets: GPU FPS>=25, latency<=40ms, GPU<=2GB) ===")
        print(f"avg FPS         : {avg_fps:.1f}   (target >= 25)")
        print(f"avg latency/frame: {avg_lat:.1f} ms (target <= 40)")
        print(f"peak GPU memory : {peak:.0f} MB  (target <= 2048)")


if __name__ == "__main__":
    main()
