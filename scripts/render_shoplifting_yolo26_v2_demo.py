"""Render the opened-development YOLO26 + MViT + R3D18 shoplifting demo.

This renderer is intentionally bound to one frozen test example and two
existing prediction artifacts:

* source: ``shoplifting/shoplifting-47.mp4`` (ground truth Shoplifting);
* YOLO-person/bag-tube MViT video probability: 0.8370326757;
* R3D18 five-seed safety-ensemble probability: 0.2058638632;
* final video risk: ``max(MViT, R3D18)`` = 0.8370326757.

Before rendering, the script joins both complete test prediction sets and
recomputes the final max-ensemble metrics. Rendering stops unless the frozen
30-video result is accuracy 93.3%, balanced accuracy 90.0%, and recall 100%.

Every source frame is passed through YOLO26s + ByteTrack for person, backpack,
handbag, and suitcase perception. Dynamic MViT and R3D18 scores are interpolated
from their recorded clip timelines. The MViT person/bag tube is interpolated
from the recorded crop boxes and drawn independently from the live tracker.

Output is transcoded to H.264/yuv420p, 640x480 at 30 fps, decoded completely by
FFmpeg and OpenCV, and sampled again from the final file for the JPEG preview.

Run from the repository root:

    .venv-yolo\\Scripts\\python.exe scripts\\render_shoplifting_yolo26_v2_demo.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import imageio_ffmpeg
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))

from ultralytics import YOLO  # noqa: E402


TARGET_RELATIVE_PATH = "shoplifting/shoplifting-47.mp4"
DEFAULT_INPUT = ROOT / "data" / "shoplifting-video-dataset" / TARGET_RELATIVE_PATH
DEFAULT_MVIT_PREDICTIONS = (
    ROOT
    / "docs"
    / "results"
    / "shoplifting-yolo26-v2"
    / "yolo-mvit-predictions.json"
)
DEFAULT_R3D_PREDICTIONS = (
    ROOT / "docs" / "results" / "shoplifting" / "predictions.json"
)
DEFAULT_YOLO_MODEL = ROOT / "models" / "yolo26s.pt"
DEFAULT_OUTPUT = (
    ROOT / "public" / "assets" / "video" / "shoplifting-yolo26-v2-demo.mp4"
)
DEFAULT_PREVIEW = (
    ROOT
    / "docs"
    / "results"
    / "shoplifting-yolo26-v2"
    / "shoplifting-yolo26-v2-demo-preview.jpg"
)

EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480
EXPECTED_FPS = 30.0
EXPECTED_FRAMES = 325
EXPECTED_MVIT_VIDEO_SCORE = 0.837032675743103
EXPECTED_R3D_VIDEO_SCORE = 0.2058638632297516
EXPECTED_FINAL_VIDEO_SCORE = EXPECTED_MVIT_VIDEO_SCORE
EXPECTED_TEST_ACCURACY = 28 / 30
EXPECTED_TEST_BALANCED_ACCURACY = 0.90
EXPECTED_TEST_RECALL = 1.0
FIXED_THRESHOLD = 0.50
DATASET_DOI = "10.17632/r3yjf35hzr.1"
AUTHOR = "Edcosys"

PERSON_CLASS = 0
BAG_CLASSES = (24, 26, 28)  # COCO backpack, handbag, suitcase
TRACK_CLASSES = (PERSON_CLASS, *BAG_CLASSES)

INK = (18, 23, 29)
WHITE = (245, 248, 250)
MUTED = (184, 199, 207)
TEAL = (182, 213, 39)
MVIT_COLOR = (237, 192, 66)
R3D_COLOR = (73, 186, 244)
FINAL_COLOR = (81, 79, 246)
TUBE_COLOR = (235, 106, 202)
BAG_COLOR = (63, 180, 255)
GREEN = (102, 207, 92)


@dataclass(frozen=True)
class PredictionBundle:
    mvit_target: dict[str, Any]
    r3d_target: dict[str, Any]
    ensemble_metrics: dict[str, Any]


@dataclass
class RenderStats:
    frames_written: int
    inference_ms: list[float]
    people_per_frame: list[int]
    bags_per_frame: list[int]
    person_track_ids: set[int]
    bag_track_ids: set[int]
    preview_frame_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the opened-development YOLO26/MViT/R3D18 v2 demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--mvit-predictions", type=Path, default=DEFAULT_MVIT_PREDICTIONS
    )
    parser.add_argument(
        "--r3d-predictions", type=Path, default=DEFAULT_R3D_PREDICTIONS
    )
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--confidence", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--preview-frame",
        type=int,
        default=292,
        help="Frame decoded from the final H.264 output for the preview.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no predictions list")
    return rows


def find_target(rows: Sequence[dict[str, Any]], artifact: Path) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if str(row.get("relative_path", "")).casefold()
        == TARGET_RELATIVE_PATH.casefold()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {TARGET_RELATIVE_PATH} row in {artifact}; found {len(matches)}"
        )
    row = matches[0]
    if row.get("split") != "test":
        raise RuntimeError(f"Target row is not test in {artifact}")
    if int(row.get("ground_truth", -1)) != 1:
        raise RuntimeError(f"Target ground truth is not Shoplifting in {artifact}")
    return row


def binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    predicted = (scores >= FIXED_THRESHOLD).astype(np.int64)
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "support": int(truth.size),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / truth.size if truth.size else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "recall": recall,
        "specificity": specificity,
    }


def validate_prediction_bundle(
    mvit_path: Path,
    r3d_path: Path,
) -> PredictionBundle:
    mvit_rows = read_prediction_rows(mvit_path)
    r3d_rows = read_prediction_rows(r3d_path)
    mvit_target = find_target(mvit_rows, mvit_path)
    r3d_target = find_target(r3d_rows, r3d_path)

    mvit_test = {
        str(row["relative_path"]).casefold(): row
        for row in mvit_rows
        if row.get("split") == "test"
    }
    r3d_test = {
        str(row["relative_path"]).casefold(): row
        for row in r3d_rows
        if row.get("split") == "test"
    }
    common = sorted(set(mvit_test) & set(r3d_test))
    if len(common) != 30:
        raise RuntimeError(f"Expected 30 joined test predictions; found {len(common)}")
    labels: list[int] = []
    final_scores: list[float] = []
    for key in common:
        mvit = mvit_test[key]
        r3d = r3d_test[key]
        if int(mvit["ground_truth"]) != int(r3d["ground_truth"]):
            raise RuntimeError(f"Ground-truth mismatch for {mvit['relative_path']}")
        labels.append(int(mvit["ground_truth"]))
        final_scores.append(
            max(
                float(mvit["shoplifting_probability"]),
                float(r3d["shoplifting_probability"]),
            )
        )
    metrics = binary_metrics(labels, final_scores)
    expected = {
        "support": 30,
        "accuracy": EXPECTED_TEST_ACCURACY,
        "balanced_accuracy": EXPECTED_TEST_BALANCED_ACCURACY,
        "recall": EXPECTED_TEST_RECALL,
    }
    for name, value in expected.items():
        if not math.isclose(float(metrics[name]), float(value), abs_tol=1e-12):
            raise RuntimeError(
                f"Frozen final-ensemble {name} changed: {metrics[name]} != {value}"
            )

    mvit_score = float(mvit_target["shoplifting_probability"])
    r3d_score = float(r3d_target["shoplifting_probability"])
    final_score = max(mvit_score, r3d_score)
    for name, actual, expected_value in (
        ("MViT", mvit_score, EXPECTED_MVIT_VIDEO_SCORE),
        ("R3D18", r3d_score, EXPECTED_R3D_VIDEO_SCORE),
        ("final", final_score, EXPECTED_FINAL_VIDEO_SCORE),
    ):
        if not math.isclose(actual, expected_value, abs_tol=1e-9):
            raise RuntimeError(
                f"Target {name} score changed: {actual} != {expected_value}"
            )
    return PredictionBundle(mvit_target, r3d_target, metrics)


def inspect_source(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    metadata = {
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    capture.release()
    expected = {
        "width": EXPECTED_WIDTH,
        "height": EXPECTED_HEIGHT,
        "frames": EXPECTED_FRAMES,
    }
    for key, value in expected.items():
        if metadata[key] != value:
            raise RuntimeError(f"Unexpected source {key}: {metadata[key]} != {value}")
    if not math.isclose(metadata["fps"], EXPECTED_FPS, abs_tol=0.01):
        raise RuntimeError(f"Unexpected source fps: {metadata['fps']}")
    return metadata


def interpolate_timelines(
    bundle: PredictionBundle,
    frame_count: int,
    clip_length: int = 16,
    frame_stride: int = 4,
) -> dict[str, np.ndarray]:
    frames = np.arange(frame_count, dtype=np.float64)
    mvit_starts = np.asarray(
        bundle.mvit_target["clip_start_frames"], dtype=np.float64
    )
    mvit_scores = np.asarray(
        bundle.mvit_target["clip_shoplifting_probabilities"], dtype=np.float64
    )
    mvit_centers = np.clip(
        mvit_starts + ((clip_length - 1) * frame_stride) / 2.0,
        0,
        frame_count - 1,
    )
    r3d_fractions = np.asarray(
        bundle.r3d_target["clip_center_fractions"], dtype=np.float64
    )
    r3d_centers = r3d_fractions * (frame_count - 1)
    r3d_scores = np.asarray(
        bundle.r3d_target["clip_shoplifting_probabilities"], dtype=np.float64
    )
    if mvit_centers.size != mvit_scores.size:
        raise RuntimeError("MViT clip timeline length mismatch")
    if r3d_centers.size != r3d_scores.size:
        raise RuntimeError("R3D18 clip timeline length mismatch")
    mvit_dynamic = np.interp(frames, mvit_centers, mvit_scores)
    r3d_dynamic = np.interp(frames, r3d_centers, r3d_scores)
    final_dynamic = np.maximum(mvit_dynamic, r3d_dynamic)

    crop_boxes = np.asarray(bundle.mvit_target["crop_boxes_xyxy"], dtype=np.float64)
    if crop_boxes.shape != (mvit_centers.size, 4):
        raise RuntimeError(f"Unexpected crop-box shape: {crop_boxes.shape}")
    interpolated_boxes = np.column_stack(
        [np.interp(frames, mvit_centers, crop_boxes[:, index]) for index in range(4)]
    )
    person_found = np.asarray(bundle.mvit_target["person_found"], dtype=bool)
    nearest_indices = np.abs(frames[:, None] - mvit_centers[None, :]).argmin(axis=1)
    tube_visible = person_found[nearest_indices]
    return {
        "mvit": mvit_dynamic,
        "r3d": r3d_dynamic,
        "final": final_dynamic,
        "tube_boxes": interpolated_boxes,
        "tube_visible": tube_visible,
    }


def track_color(track_id: int, is_bag: bool) -> tuple[int, int, int]:
    if is_bag:
        palette = [BAG_COLOR, (79, 148, 255), (90, 211, 245), (140, 120, 255)]
    else:
        palette = [TEAL, (209, 229, 65), (175, 224, 91), (218, 201, 54)]
    return palette[abs(track_id) % len(palette)]


def draw_text_box(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.40,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - height - baseline - 5)
    cv2.rectangle(frame, (x, top), (x + width + 8, y), color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        text,
        (x + 4, y - baseline - 2),
        font,
        scale,
        INK,
        thickness,
        cv2.LINE_AA,
    )


def dashed_line(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash: int = 7,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = int(round(math.hypot(x2 - x1, y2 - y1)))
    if length <= 0:
        return
    for offset in range(0, length, dash * 2):
        first = offset / length
        second = min(1.0, (offset + dash) / length)
        p1 = (int(round(x1 + (x2 - x1) * first)), int(round(y1 + (y2 - y1) * first)))
        p2 = (
            int(round(x1 + (x2 - x1) * second)),
            int(round(y1 + (y2 - y1) * second)),
        )
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def dashed_rectangle(
    frame: np.ndarray,
    box: Sequence[int],
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = (int(value) for value in box)
    dashed_line(frame, (x1, y1), (x2, y1), color, 2)
    dashed_line(frame, (x2, y1), (x2, y2), color, 2)
    dashed_line(frame, (x2, y2), (x1, y2), color, 2)
    dashed_line(frame, (x1, y2), (x1, y1), color, 2)


def draw_perception(
    frame: np.ndarray,
    result: Any,
    person_history: dict[int, deque[tuple[int, int]]],
) -> tuple[int, int, set[int], set[int]]:
    people = 0
    bags = 0
    person_ids: set[int] = set()
    bag_ids: set[int] = set()
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return people, bags, person_ids, bag_ids
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    if boxes.id is not None:
        track_ids = boxes.id.detach().cpu().numpy().astype(int)
    else:
        track_ids = np.arange(len(xyxy), dtype=int)
    for box, confidence, class_id, track_id in zip(
        xyxy, confidences, classes, track_ids
    ):
        is_bag = class_id in BAG_CLASSES
        if class_id == PERSON_CLASS:
            people += 1
            person_ids.add(int(track_id))
            object_name = "Person"
        elif is_bag:
            bags += 1
            bag_ids.add(int(track_id))
            object_name = {24: "Backpack", 26: "Handbag", 28: "Suitcase"}.get(
                class_id, "Bag"
            )
        else:
            continue
        x1, y1, x2, y2 = (
            int(round(float(value)))
            for value in np.clip(
                box,
                [0, 0, 0, 0],
                [EXPECTED_WIDTH - 1, EXPECTED_HEIGHT - 1] * 2,
            )
        )
        color = track_color(int(track_id), is_bag)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        draw_text_box(
            frame,
            f"{object_name} #{int(track_id):02d} {float(confidence):.2f}",
            x1,
            max(18, y1),
            color,
        )
        if not is_bag:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            person_history[int(track_id)].append(center)
            points = np.asarray(person_history[int(track_id)], dtype=np.int32)
            if len(points) > 1:
                cv2.polylines(
                    frame,
                    [points.reshape((-1, 1, 2))],
                    False,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    return people, bags, person_ids, bag_ids


def draw_risk_bar(
    frame: np.ndarray,
    probability: float,
    top: int,
    color: tuple[int, int, int],
) -> None:
    left, right = 390, 624
    height = 7
    cv2.rectangle(frame, (left, top), (right, top + height), (74, 84, 92), -1)
    fill = left + int(round((right - left) * np.clip(probability, 0.0, 1.0)))
    cv2.rectangle(frame, (left, top), (fill, top + height), color, -1)
    threshold_x = left + int(round((right - left) * FIXED_THRESHOLD))
    cv2.line(
        frame,
        (threshold_x, top - 2),
        (threshold_x, top + height + 2),
        WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    frame_index: int,
    frame_count: int,
    people: int,
    bags: int,
    mvit_dynamic: float,
    r3d_dynamic: float,
    final_dynamic: float,
    bundle: PredictionBundle,
) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 151), INK, -1)
    cv2.rectangle(overlay, (0, height - 70), (width, height), INK, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (0, 0), (7, 151), TEAL, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        "SHOPLIFTING RISK DEMO | OPENED DEVELOPMENT TEST",
        (16, 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.57,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    rows = [
        (
            "YOLO26 + ByteTrack perception",
            f"people {people} | bags {bags}",
            TEAL,
        ),
        (
            "Full-frame + person/bag tube MViT",
            f"dynamic {mvit_dynamic * 100:5.1f}% | video {EXPECTED_MVIT_VIDEO_SCORE * 100:4.1f}%",
            MVIT_COLOR,
        ),
        (
            "R3D18 safety ensemble",
            f"dynamic {r3d_dynamic * 100:5.1f}% | video {EXPECTED_R3D_VIDEO_SCORE * 100:4.1f}%",
            R3D_COLOR,
        ),
        (
            "Final risk = max",
            f"dynamic {final_dynamic * 100:5.1f}% | video {EXPECTED_FINAL_VIDEO_SCORE * 100:4.1f}%",
            FINAL_COLOR,
        ),
    ]
    for row_index, (label, value, color) in enumerate(rows):
        y = 48 + row_index * 24
        cv2.circle(frame, (19, y - 4), 4, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (30, y), font, 0.41, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, value, (390, y), font, 0.39, color, 1, cv2.LINE_AA)
    draw_risk_bar(frame, mvit_dynamic, 54, MVIT_COLOR)
    draw_risk_bar(frame, r3d_dynamic, 78, R3D_COLOR)
    draw_risk_bar(frame, final_dynamic, 102, FINAL_COLOR)

    decision = "ALERT" if EXPECTED_FINAL_VIDEO_SCORE >= FIXED_THRESHOLD else "NORMAL"
    cv2.putText(
        frame,
        f"Fixed threshold .50 | GT Shoplifting | Final decision {decision}",
        (16, 143),
        font,
        0.42,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "TEST  accuracy 93.3%  |  balanced accuracy 90.0%  |  recall 100%",
        (16, height - 46),
        cv2.FONT_HERSHEY_DUPLEX,
        0.46,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Author: {AUTHOR}  |  Dataset DOI {DATASET_DOI}",
        (16, height - 24),
        font,
        0.40,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    progress_left, progress_right = 16, width - 16
    progress_y = height - 12
    cv2.rectangle(
        frame,
        (progress_left, progress_y),
        (progress_right, progress_y + 3),
        (72, 82, 90),
        -1,
    )
    progress = frame_index / max(1, frame_count - 1)
    cv2.rectangle(
        frame,
        (progress_left, progress_y),
        (
            progress_left
            + int(round((progress_right - progress_left) * progress)),
            progress_y + 3,
        ),
        FINAL_COLOR,
        -1,
    )


def render_intermediate(
    source: Path,
    intermediate: Path,
    yolo_model: Path,
    timelines: dict[str, np.ndarray],
    bundle: PredictionBundle,
    args: argparse.Namespace,
) -> RenderStats:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source {source}")
    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        EXPECTED_FPS,
        (EXPECTED_WIDTH, EXPECTED_HEIGHT),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open intermediate writer {intermediate}")
    model = YOLO(str(yolo_model))
    person_history: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=18)
    )
    inference_ms: list[float] = []
    people_per_frame: list[int] = []
    bags_per_frame: list[int] = []
    person_track_ids: set[int] = set()
    bag_track_ids: set[int] = set()
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
                frame = cv2.resize(
                    frame,
                    (EXPECTED_WIDTH, EXPECTED_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(TRACK_CLASSES),
                conf=args.confidence,
                iou=args.iou,
                imgsz=args.imgsz,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False,
            )[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_ms.append((time.perf_counter() - started) * 1000.0)

            people, bags, current_people, current_bags = draw_perception(
                frame, result, person_history
            )
            person_track_ids.update(current_people)
            bag_track_ids.update(current_bags)
            people_per_frame.append(people)
            bags_per_frame.append(bags)

            if bool(timelines["tube_visible"][frame_index]):
                box = timelines["tube_boxes"][frame_index].round().astype(int)
                box[[0, 2]] = np.clip(box[[0, 2]], 0, EXPECTED_WIDTH - 1)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, EXPECTED_HEIGHT - 1)
                area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
                if area < EXPECTED_WIDTH * EXPECTED_HEIGHT * 0.92:
                    dashed_rectangle(frame, box, TUBE_COLOR)
                    draw_text_box(
                        frame,
                        "MViT person/bag tube",
                        int(box[0]),
                        max(18, int(box[1])),
                        TUBE_COLOR,
                        scale=0.38,
                    )

            draw_hud(
                frame,
                frame_index,
                EXPECTED_FRAMES,
                people,
                bags,
                float(timelines["mvit"][frame_index]),
                float(timelines["r3d"][frame_index]),
                float(timelines["final"][frame_index]),
                bundle,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if frame_index != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Rendered {frame_index} frames; expected {EXPECTED_FRAMES}"
        )
    return RenderStats(
        frames_written=frame_index,
        inference_ms=inference_ms,
        people_per_frame=people_per_frame,
        bags_per_frame=bags_per_frame,
        person_track_ids=person_track_ids,
        bag_track_ids=bag_track_ids,
        preview_frame_index=args.preview_frame,
    )


def transcode_h264(intermediate: Path, output: Path) -> str:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(intermediate),
            "-vf",
            "scale=640:480:flags=lanczos",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            "-metadata",
            "title=Edcosys Shoplifting YOLO26 v2 Demo",
            "-metadata",
            f"artist={AUTHOR}",
            "-metadata",
            f"comment=Dataset DOI {DATASET_DOI}",
            str(output),
        ],
        check=True,
    )
    return ffmpeg


def validate_and_extract_preview(
    output: Path,
    preview: Path,
    preview_frame: int,
    ffmpeg: str,
) -> dict[str, Any]:
    decode = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(output),
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if decode.returncode != 0:
        raise RuntimeError(f"FFmpeg full decode failed:\n{decode.stderr}")
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stderr
    probe_casefold = probe.casefold()
    if "h264" not in probe_casefold or "yuv420p" not in probe_casefold:
        raise RuntimeError(f"Output is not H.264/yuv420p:\n{probe}")

    capture = cv2.VideoCapture(str(output))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open final output {output}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    decoded_frames = 0
    preview_image: np.ndarray | None = None
    luminance_means: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
            capture.release()
            raise RuntimeError(f"Decoded frame has wrong shape: {frame.shape}")
        if not np.isfinite(frame).all():
            capture.release()
            raise RuntimeError(f"Non-finite decoded frame at {decoded_frames}")
        luminance_means.append(float(frame.mean()))
        if decoded_frames == preview_frame:
            preview_image = frame.copy()
        decoded_frames += 1
    capture.release()
    if decoded_frames != EXPECTED_FRAMES:
        raise RuntimeError(
            f"OpenCV decoded {decoded_frames} frames; expected {EXPECTED_FRAMES}"
        )
    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        raise RuntimeError(f"Output dimensions are {width}x{height}")
    if not math.isclose(fps, EXPECTED_FPS, abs_tol=0.01):
        raise RuntimeError(f"Output fps is {fps}")
    if preview_image is None:
        raise RuntimeError(f"Preview frame {preview_frame} is outside the output")
    if statistics.pstdev(luminance_means) < 1.0:
        raise RuntimeError("Decoded output appears visually static")
    preview.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(preview), preview_image, [cv2.IMWRITE_JPEG_QUALITY, 94]
    ):
        raise RuntimeError(f"Could not write preview {preview}")
    return {
        "codec_probe_contains_h264": True,
        "pixel_format_probe_contains_yuv420p": True,
        "width": width,
        "height": height,
        "fps": fps,
        "decoded_frames": decoded_frames,
        "duration_seconds": decoded_frames / fps,
        "preview_frame": preview_frame,
        "preview_mean": float(preview_image.mean()),
        "preview_std": float(preview_image.std()),
        "video_luminance_std": statistics.pstdev(luminance_means),
    }


def main() -> int:
    args = parse_args()
    args.input = args.input.resolve()
    args.mvit_predictions = args.mvit_predictions.resolve()
    args.r3d_predictions = args.r3d_predictions.resolve()
    args.yolo_model = args.yolo_model.resolve()
    args.output = args.output.resolve()
    args.preview = args.preview.resolve()
    if not 0 < args.confidence <= 1:
        raise ValueError("--confidence must be in (0, 1]")
    if not 0 < args.iou <= 1:
        raise ValueError("--iou must be in (0, 1]")
    if not 0 <= args.preview_frame < EXPECTED_FRAMES:
        raise ValueError(f"--preview-frame must be in [0, {EXPECTED_FRAMES - 1}]")
    if not args.yolo_model.is_file():
        raise FileNotFoundError(args.yolo_model)
    if not torch.cuda.is_available():
        raise RuntimeError("The fixed demo renderer requires the local CUDA GPU")

    source_metadata = inspect_source(args.input)
    bundle = validate_prediction_bundle(
        args.mvit_predictions, args.r3d_predictions
    )
    timelines = interpolate_timelines(bundle, EXPECTED_FRAMES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()
    temporary_root = ROOT / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="shoplifting-yolo26-v2-", dir=temporary_root
    ) as temporary_directory:
        intermediate = Path(temporary_directory) / "intermediate-mp4v.mp4"
        stats = render_intermediate(
            args.input,
            intermediate,
            args.yolo_model,
            timelines,
            bundle,
            args,
        )
        ffmpeg = transcode_h264(intermediate, args.output)

    validation = validate_and_extract_preview(
        args.output, args.preview, args.preview_frame, ffmpeg
    )
    elapsed = time.perf_counter() - wall_start
    summary = {
        "source": {
            "relative_path": TARGET_RELATIVE_PATH,
            "sha256": sha256_file(args.input),
            **source_metadata,
            "ground_truth": "Shoplifting",
            "split": "test",
        },
        "predictions": {
            "mvit_video_probability": EXPECTED_MVIT_VIDEO_SCORE,
            "r3d18_video_probability": EXPECTED_R3D_VIDEO_SCORE,
            "final_rule": "max(MViT, R3D18)",
            "final_video_probability": EXPECTED_FINAL_VIDEO_SCORE,
            "threshold": FIXED_THRESHOLD,
        },
        "test_metrics": bundle.ensemble_metrics,
        "perception": {
            "model": args.yolo_model.name,
            "model_sha256": sha256_file(args.yolo_model),
            "tracker": "ByteTrack",
            "classes": ["person", "backpack", "handbag", "suitcase"],
            "confidence": args.confidence,
            "iou": args.iou,
            "person_track_ids": len(stats.person_track_ids),
            "bag_track_ids": len(stats.bag_track_ids),
            "mean_people_per_frame": statistics.fmean(stats.people_per_frame),
            "mean_bags_per_frame": statistics.fmean(stats.bags_per_frame),
            "inference_ms_mean": statistics.fmean(stats.inference_ms),
            "inference_ms_p95": float(np.percentile(stats.inference_ms, 95)),
        },
        "output": {
            "video": str(args.output.relative_to(ROOT)),
            "video_sha256": sha256_file(args.output),
            "video_bytes": args.output.stat().st_size,
            "preview": str(args.preview.relative_to(ROOT)),
            "preview_sha256": sha256_file(args.preview),
            "preview_bytes": args.preview.stat().st_size,
            "encoding": "H.264 yuv420p",
            **validation,
        },
        "attribution": {
            "author": AUTHOR,
            "dataset_doi": DATASET_DOI,
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "frames_rendered": stats.frames_written,
            "wall_seconds": elapsed,
            "pipeline_fps": stats.frames_written / elapsed,
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
