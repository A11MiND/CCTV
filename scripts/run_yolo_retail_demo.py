"""Run a reproducible YOLO26 + ByteTrack retail perception demo.

This script intentionally demonstrates person detection and tracking only.  It
does not infer intent or label anyone as shoplifting.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE))

from ultralytics import YOLO  # noqa: E402


TEAL = (214, 203, 0)
AMBER = (0, 172, 255)
INK = (16, 21, 25)
WHITE = (245, 247, 249)


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (WORKSPACE / path).resolve()


def source_metadata(input_path: Path) -> dict[str, str]:
    if "shoplifting-video-dataset" in input_path.as_posix().lower():
        return {
            "label": "Dataset DOI 10.17632/r3yjf35hzr.1",
            "page": "https://data.mendeley.com/datasets/r3yjf35hzr/1",
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "creator": "Mohd. Aquib Ansari and Dushyant Kumar Singh",
        }
    return {
        "label": "Pexels / Suika Chan",
        "page": "https://www.pexels.com/video/customers-shopping-at-supermarket-10901926/",
        "license": "https://www.pexels.com/license/",
        "creator": "Suika Chan",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=WORKSPACE / "assets/video/pexels-hong-kong-supermarket.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "public/assets/video/yolo26-retail-demo.mp4",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=WORKSPACE / "assets/video/yolo26-run-metrics.json",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=WORKSPACE / "public/assets/video/yolo26-preview.jpg",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=WORKSPACE / "models/yolo26s.pt",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--target-fps", type=float, default=15.0)
    return parser.parse_args()


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def track_color(track_id: int) -> tuple[int, int, int]:
    palette = [
        (214, 203, 0),
        (192, 229, 45),
        (255, 181, 62),
        (214, 133, 40),
        (246, 207, 95),
    ]
    return palette[track_id % len(palette)]


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.48
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(
        frame,
        (x, y - height - baseline - 8),
        (x + width + 12, y),
        color,
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x + 6, y - baseline - 4),
        font,
        font_scale,
        INK,
        thickness,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    person_count: int,
    inference_ms: float,
    device_name: str,
    source_label: str,
) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (578, 104), INK, -1)
    cv2.rectangle(overlay, (18, height - 55), (width - 18, height - 18), INK, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (18, 18), (25, 104), TEAL, -1)
    cv2.putText(
        frame,
        "YOLO26s + ByteTrack | REAL VIDEO",
        (40, 52),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{device_name}  |  {inference_ms:.1f} ms  |  persons {person_count}",
        (40, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (204, 213, 218),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (42, height - 36), 5, AMBER, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "PERCEPTION DEMO ONLY - NOT A THEFT OR INTENT DECISION",
        (58, height - 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Source: {source_label}",
        (width - 375, height - 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (181, 190, 196),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    for attribute in ("input", "output", "metrics", "preview", "model"):
        setattr(args, attribute, resolve_path(getattr(args, attribute)))
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected; this demo is expected to run on GPU.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.input}")

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, round(source_fps / args.target_fps))
    output_fps = source_fps / frame_step
    output_width, output_height = 1280, 720
    intermediate = args.output.with_name(f"{args.output.stem}-mp4v.mp4")

    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer {intermediate}")

    model = YOLO(str(args.model))
    source = source_metadata(args.input)
    device_name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
    history: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=24))
    inference_times: list[float] = []
    per_frame_counts: list[int] = []
    unique_track_ids: set[int] = set()
    frames_read = 0
    frames_written = 0
    preview_written = False
    wall_start = time.perf_counter()

    while True:
        ok, source_frame = capture.read()
        if not ok:
            break
        current_index = frames_read
        frames_read += 1
        if current_index % frame_step:
            continue

        frame = cv2.resize(
            source_frame,
            (output_width, output_height),
            interpolation=cv2.INTER_AREA,
        )
        torch.cuda.synchronize()
        infer_start = time.perf_counter()
        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=args.conf,
            imgsz=args.imgsz,
            device=0,
            verbose=False,
        )[0]
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - infer_start) * 1000
        inference_times.append(inference_ms)

        person_count = 0
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            if boxes.id is not None:
                track_ids = boxes.id.detach().cpu().numpy().astype(int)
            else:
                track_ids = np.arange(len(xyxy), dtype=int)

            person_count = len(xyxy)
            for box, confidence, track_id in zip(xyxy, confidences, track_ids):
                unique_track_ids.add(int(track_id))
                x1, y1, x2, y2 = (int(value) for value in box)
                color = track_color(int(track_id))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                draw_label(
                    frame,
                    f"Person #{int(track_id):02d}  {confidence:.2f}",
                    (x1, max(28, y1)),
                    color,
                )
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                history[int(track_id)].append(center)
                points = np.asarray(history[int(track_id)], dtype=np.int32)
                if len(points) > 1:
                    cv2.polylines(
                        frame,
                        [points.reshape((-1, 1, 2))],
                        False,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

        per_frame_counts.append(person_count)
        draw_hud(frame, person_count, inference_ms, device_name, source["label"])
        writer.write(frame)
        frames_written += 1

        if not preview_written and frames_written >= round(output_fps * 4):
            cv2.imwrite(str(args.preview), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            preview_written = True

    capture.release()
    writer.release()
    wall_seconds = time.perf_counter() - wall_start

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(intermediate),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(args.output),
        ],
        check=True,
    )
    intermediate.unlink(missing_ok=True)

    metrics = {
        "run_type": "real-video person detection and tracking demo",
        "limitation": "Not a shoplifting, criminal-intent, or identity decision.",
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "page": source["page"],
            "license": source["license"],
            "creator": source["creator"],
            "input_file": str(args.input.relative_to(WORKSPACE)),
            "resolution": [source_width, source_height],
            "fps": source_fps,
            "frames": source_frames,
            "duration_seconds": round(source_frames / source_fps, 3),
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_available": torch.cuda.is_available(),
            "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "model": args.model.name,
            "tracker": "ByteTrack",
            "imgsz": args.imgsz,
            "confidence_threshold": args.conf,
            "fp16": False,
        },
        "result": {
            "frames_processed": frames_written,
            "output_fps": output_fps,
            "wall_seconds": round(wall_seconds, 3),
            "pipeline_fps_including_render": round(frames_written / wall_seconds, 2),
            "inference_ms_mean": round(statistics.fmean(inference_times), 2),
            "inference_ms_p50": round(percentile(inference_times, 50), 2),
            "inference_ms_p95": round(percentile(inference_times, 95), 2),
            "persons_per_frame_mean": round(statistics.fmean(per_frame_counts), 2),
            "persons_per_frame_max": max(per_frame_counts, default=0),
            "unique_track_ids": len(unique_track_ids),
            "output_file": str(args.output.relative_to(WORKSPACE)),
        },
    }
    args.metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
