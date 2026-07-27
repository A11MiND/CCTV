"""YOLO-guided dual-view MViT MIL experiment.

YOLO26s participates directly in this classifier by defining a person-tube
view for every temporal clip:

1. Detect COCO person/backpack/handbag/suitcase classes on the clip's center
   frame.
2. Select the largest visible person with a fixed label-free rule.
3. Union nearby bags with that person and expand the result by 1.6x.
4. Apply the same crop to all 16 frames in the clip.
5. Extract a frozen 768-D MViT-V2-S Kinetics-400 embedding from the crop.

The experiment reuses the 21-window full-frame cache produced by
``shoplifting_mvit_experiment.py`` and compares full-frame, crop-only, and
dual-view MIL heads. Pooling and dual fusion are selected using validation
metrics only. ``run`` and default ``evaluate`` process train/validation rows
only. The fixed internal test split is opened only by:

    .venv-yolo\\Scripts\\python.exe \
      scripts\\shoplifting_yolo_mvit_experiment.py evaluate --include-test

Primary outputs:

    output/shoplifting-yolo-mvit-experiment/summary.json
    output/shoplifting-yolo-mvit-experiment/checkpoint.pt
    output/shoplifting-yolo-mvit-experiment/predictions.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import torch
from torch import nn

import shoplifting_mil_baseline as baseline
import shoplifting_mvit_experiment as mvit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "shoplifting-video-dataset"
DEFAULT_MANIFEST = (
    ROOT / "docs" / "results" / "shoplifting" / "actor-clothing-disjoint-split.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "output" / "shoplifting-yolo-mvit-experiment"
DEFAULT_YOLO_MODEL = ROOT / "models" / "yolo26s.pt"
DEFAULT_FULL_CACHE = mvit.default_cache_path(21)
DEFAULT_CROP_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "yolo26s-mvit-v2-s-person-tube-21-embeddings.npz"
)

CLIPS_PER_VIDEO = 21
FRAMES_PER_CLIP = 16
FRAME_STRIDE = 4
EMBEDDING_DIMENSION = 768
HIDDEN_DIMENSION = 128
FIXED_THRESHOLD = 0.50
DEFAULT_SEEDS = (11, 23, 37, 51, 79)
POOLING_CHOICES = ("top1", "top2", "logsumexp", "attention")
DUAL_FUSION_CHOICES = ("concat", "gated")
YOLO_CLASSES = (0, 24, 26, 28)
PERSON_CLASS = 0
ACCESSORY_CLASSES = frozenset((24, 26, 28))
ACCESSORY_NAMES = {24: "backpack", 26: "handbag", 28: "suitcase"}
CROP_EXPANSION = 1.60
SCRIPT_VERSION = "1.0.1"

if (ROOT / "data" / "torch-cache").is_dir():
    os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "torch-cache"))


class DualViewMILHead(nn.Module):
    """Full, crop, or fused clip logits followed by MIL pooling."""

    def __init__(
        self,
        view: str,
        pooling: str,
        fusion: str = "none",
        dropout: float = 0.30,
        logsumexp_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if view not in {"full", "crop", "dual"}:
            raise ValueError(f"Unsupported view: {view}")
        if pooling not in POOLING_CHOICES:
            raise ValueError(f"Unsupported pooling: {pooling}")
        if view == "dual" and fusion not in DUAL_FUSION_CHOICES:
            raise ValueError(f"Unsupported dual fusion: {fusion}")
        if view != "dual" and fusion != "none":
            raise ValueError("Single-view heads require fusion='none'")
        if logsumexp_temperature <= 0:
            raise ValueError("logsumexp_temperature must be positive")
        self.view = view
        self.pooling = pooling
        self.fusion = fusion
        self.logsumexp_temperature = float(logsumexp_temperature)

        def classifier(input_dimension: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dimension, HIDDEN_DIMENSION),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(HIDDEN_DIMENSION, 1),
            )

        if view in {"full", "crop"}:
            self.single_classifier = classifier(EMBEDDING_DIMENSION)
            self.concat_projection = None
            self.concat_classifier = None
            self.full_classifier = None
            self.crop_classifier = None
            self.gate = None
            attention_dimension = EMBEDDING_DIMENSION
        elif fusion == "concat":
            self.single_classifier = None
            self.concat_projection = nn.Sequential(
                nn.LayerNorm(EMBEDDING_DIMENSION * 2),
                nn.Linear(EMBEDDING_DIMENSION * 2, HIDDEN_DIMENSION * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.concat_classifier = nn.Linear(HIDDEN_DIMENSION * 2, 1)
            self.full_classifier = None
            self.crop_classifier = None
            self.gate = None
            attention_dimension = HIDDEN_DIMENSION * 2
        else:
            self.single_classifier = None
            self.concat_projection = None
            self.concat_classifier = None
            self.full_classifier = classifier(EMBEDDING_DIMENSION)
            self.crop_classifier = classifier(EMBEDDING_DIMENSION)
            self.gate = nn.Sequential(
                nn.LayerNorm(EMBEDDING_DIMENSION * 2),
                nn.Linear(EMBEDDING_DIMENSION * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            attention_dimension = EMBEDDING_DIMENSION
        self.attention = (
            nn.Sequential(
                nn.Linear(attention_dimension, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            if pooling == "attention"
            else None
        )

    def clip_logits_and_representation(
        self,
        full_embeddings: torch.Tensor,
        crop_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.view == "full":
            assert self.single_classifier is not None
            return (
                self.single_classifier(full_embeddings).squeeze(-1),
                full_embeddings,
            )
        if self.view == "crop":
            assert self.single_classifier is not None
            return (
                self.single_classifier(crop_embeddings).squeeze(-1),
                crop_embeddings,
            )
        concatenated = torch.cat((full_embeddings, crop_embeddings), dim=-1)
        if self.fusion == "concat":
            assert self.concat_projection is not None
            assert self.concat_classifier is not None
            representation = self.concat_projection(concatenated)
            return self.concat_classifier(representation).squeeze(-1), representation
        assert self.full_classifier is not None
        assert self.crop_classifier is not None
        assert self.gate is not None
        gate = torch.sigmoid(self.gate(concatenated))
        full_logits = self.full_classifier(full_embeddings)
        crop_logits = self.crop_classifier(crop_embeddings)
        logits = ((1.0 - gate) * full_logits + gate * crop_logits).squeeze(-1)
        representation = (
            (1.0 - gate) * full_embeddings + gate * crop_embeddings
        )
        return logits, representation

    def clip_logits(
        self,
        full_embeddings: torch.Tensor,
        crop_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.clip_logits_and_representation(
            full_embeddings, crop_embeddings
        )[0]

    def forward(
        self,
        full_embeddings: torch.Tensor,
        crop_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        logits, representation = self.clip_logits_and_representation(
            full_embeddings, crop_embeddings
        )
        if self.pooling == "top1":
            return logits.max(dim=1).values
        if self.pooling == "top2":
            return logits.topk(min(2, logits.shape[1]), dim=1).values.mean(dim=1)
        if self.pooling == "logsumexp":
            temperature = self.logsumexp_temperature
            return temperature * (
                torch.logsumexp(logits / temperature, dim=1)
                - math.log(logits.shape[1])
            )
        assert self.attention is not None
        weights = torch.softmax(self.attention(representation).squeeze(-1), dim=1)
        return (weights * logits).sum(dim=1)


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def clip_box(
    box: Sequence[float], frame_width: int, frame_height: int
) -> tuple[float, float, float, float]:
    x1 = float(np.clip(box[0], 0, max(0, frame_width - 1)))
    y1 = float(np.clip(box[1], 0, max(0, frame_height - 1)))
    x2 = float(np.clip(box[2], x1 + 1, frame_width))
    y2 = float(np.clip(box[3], y1 + 1, frame_height))
    return x1, y1, x2, y2


def boxes_are_near(
    person: Sequence[float], accessory: Sequence[float]
) -> bool:
    px1, py1, px2, py2 = map(float, person)
    ax1, ay1, ax2, ay2 = map(float, accessory)
    person_width = max(1.0, px2 - px1)
    person_height = max(1.0, py2 - py1)
    accessory_center = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)
    margin_x = 0.35 * person_width
    margin_y = 0.35 * person_height
    center_in_expanded_person = (
        px1 - margin_x <= accessory_center[0] <= px2 + margin_x
        and py1 - margin_y <= accessory_center[1] <= py2 + margin_y
    )
    horizontal_gap = max(px1 - ax2, ax1 - px2, 0.0)
    vertical_gap = max(py1 - ay2, ay1 - py2, 0.0)
    edge_distance = math.hypot(horizontal_gap, vertical_gap)
    return center_in_expanded_person or edge_distance <= 0.25 * max(
        person_width, person_height
    )


def expand_box(
    box: Sequence[float],
    frame_width: int,
    frame_height: int,
    factor: float = CROP_EXPANSION,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = clip_box(box, frame_width, frame_height)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = max(32.0, (x2 - x1) * factor)
    height = max(32.0, (y2 - y1) * factor)
    expanded = clip_box(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        frame_width,
        frame_height,
    )
    integer_box = (
        int(math.floor(expanded[0])),
        int(math.floor(expanded[1])),
        int(math.ceil(expanded[2])),
        int(math.ceil(expanded[3])),
    )
    if integer_box[2] <= integer_box[0] or integer_box[3] <= integer_box[1]:
        return 0, 0, frame_width, frame_height
    return integer_box


def select_label_free_crop(
    detections: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[tuple[int, int, int, int], bool, bool, list[int]]:
    """Select the largest visible person and union nearby COCO bag classes."""

    if detections.size == 0:
        return (0, 0, frame_width, frame_height), False, False, []
    people: list[tuple[float, float, float, float, float, int]] = []
    accessories: list[tuple[float, float, float, float, float, int]] = []
    for raw in detections:
        x1, y1, x2, y2, confidence, class_id_raw = map(float, raw[:6])
        class_id = int(round(class_id_raw))
        clipped = clip_box((x1, y1, x2, y2), frame_width, frame_height)
        item = (*clipped, confidence, class_id)
        if class_id == PERSON_CLASS:
            people.append(item)
        elif class_id in ACCESSORY_CLASSES:
            accessories.append(item)
    if not people:
        return (0, 0, frame_width, frame_height), False, False, []
    # Area and confidence are label-free visibility signals.
    selected = max(
        people,
        key=lambda item: (box_area(item[:4]), item[4], -item[0], -item[1]),
    )
    union = list(selected[:4])
    included_classes: list[int] = []
    for accessory in accessories:
        if boxes_are_near(selected[:4], accessory[:4]):
            union[0] = min(union[0], accessory[0])
            union[1] = min(union[1], accessory[1])
            union[2] = max(union[2], accessory[2])
            union[3] = max(union[3], accessory[3])
            included_classes.append(accessory[5])
    return (
        expand_box(union, frame_width, frame_height, CROP_EXPANSION),
        True,
        bool(included_classes),
        sorted(included_classes),
    )


def yolo_device_argument(device: torch.device) -> str | int:
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    return "cpu"


def detect_video_crops(
    video_path: Path,
    starts: np.ndarray,
    yolo_model: Any,
    device: torch.device,
    confidence: float,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frame_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    frame_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    center_offset = (FRAMES_PER_CLIP // 2) * FRAME_STRIDE
    center_frames: list[np.ndarray] = []
    try:
        for start in starts:
            center_index = min(frame_count - 1, int(start) + center_offset)
            capture.set(cv2.CAP_PROP_POS_FRAMES, center_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Could not decode center frame {center_index} from {video_path}"
                )
            center_frames.append(frame)
    finally:
        capture.release()
    results = yolo_model.predict(
        source=center_frames,
        classes=list(YOLO_CLASSES),
        conf=confidence,
        imgsz=image_size,
        device=yolo_device_argument(device),
        verbose=False,
        stream=False,
    )
    if len(results) != len(starts):
        raise RuntimeError("YOLO returned an unexpected result count")
    crop_boxes: list[tuple[int, int, int, int]] = []
    person_found: list[bool] = []
    accessory_union: list[bool] = []
    accessory_classes: list[list[int]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None or not len(boxes):
            detections = np.empty((0, 6), dtype=np.float32)
        else:
            detections = np.column_stack(
                (
                    boxes.xyxy.detach().cpu().numpy(),
                    boxes.conf.detach().cpu().numpy(),
                    boxes.cls.detach().cpu().numpy(),
                )
            )
        crop, found, unioned, included = select_label_free_crop(
            detections, frame_width, frame_height
        )
        crop_boxes.append(crop)
        person_found.append(found)
        accessory_union.append(unioned)
        accessory_classes.append(included)
    return (
        np.asarray(crop_boxes, dtype=np.int64),
        np.asarray(person_found, dtype=np.bool_),
        np.asarray(accessory_union, dtype=np.bool_),
        accessory_classes,
    )


def iter_person_tube_clips(
    video_path: Path,
    starts: np.ndarray,
    crop_boxes: np.ndarray,
    preprocess: nn.Module,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Decode once and apply each center-frame crop to all 16 clip frames."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    targets: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for clip_index, start in enumerate(starts.tolist()):
        for slot in range(FRAMES_PER_CLIP):
            frame_index = min(
                frame_count - 1, int(start) + slot * FRAME_STRIDE
            )
            targets[frame_index].append((clip_index, slot))
    buffers: list[list[np.ndarray | None] | None] = [
        [None] * FRAMES_PER_CLIP for _ in starts
    ]
    completed: set[int] = set()
    try:
        for frame_index in range(max(targets) + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Could not decode frame {frame_index} from {video_path}"
                )
            assignments = targets.get(frame_index)
            if not assignments:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            touched: set[int] = set()
            for clip_index, slot in assignments:
                buffer = buffers[clip_index]
                if buffer is None:
                    continue
                x1, y1, x2, y2 = crop_boxes[clip_index]
                crop = rgb[int(y1) : int(y2), int(x1) : int(x2)]
                if crop.size == 0:
                    crop = rgb
                buffer[slot] = crop
                touched.add(clip_index)
            for clip_index in sorted(touched):
                buffer = buffers[clip_index]
                if buffer is None or any(item is None for item in buffer):
                    continue
                # Crop dimensions are constant within a clip because one center
                # frame box is applied to all frames.
                tensor = (
                    torch.from_numpy(np.stack(buffer))
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                buffers[clip_index] = None
                completed.add(clip_index)
                yield clip_index, preprocess(tensor)
    finally:
        capture.release()
    if len(completed) != len(starts):
        missing = sorted(set(range(len(starts))) - completed)
        raise RuntimeError(f"Incomplete person-tube clips in {video_path}: {missing}")


def load_crop_cache(
    cache_path: Path,
    yolo_sha256: str,
    confidence: float,
    image_size: int,
) -> dict[str, dict[str, Any]]:
    if not cache_path.is_file():
        return {}
    with np.load(cache_path, allow_pickle=False) as payload:
        required = {
            "features",
            "labels",
            "paths",
            "splits",
            "start_frames",
            "crop_boxes",
            "person_found",
            "accessory_union",
            "yolo_sha256",
            "yolo_confidence",
            "yolo_image_size",
            "crop_expansion",
            "frames_per_clip",
            "frame_stride",
        }
        missing = required - set(payload.files)
        if missing:
            raise RuntimeError(f"Crop cache is missing arrays: {sorted(missing)}")
        cached_hash = str(np.asarray(payload["yolo_sha256"]).item())
        if cached_hash != yolo_sha256:
            raise RuntimeError(
                "Crop cache YOLO hash differs from models/yolo26s.pt; "
                "use --force-crop-extract to rebuild it"
            )
        cached_confidence = float(np.asarray(payload["yolo_confidence"]).item())
        cached_image_size = int(np.asarray(payload["yolo_image_size"]).item())
        cached_expansion = float(np.asarray(payload["crop_expansion"]).item())
        if (
            not math.isclose(cached_confidence, confidence, abs_tol=1e-9)
            or cached_image_size != image_size
            or not math.isclose(
                cached_expansion, CROP_EXPANSION, abs_tol=1e-9
            )
        ):
            raise RuntimeError(
                "Crop cache YOLO/crop settings differ from this run; "
                "use --force-crop-extract to rebuild it"
            )
        features = payload["features"].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
        paths = payload["paths"].astype(str)
        splits = payload["splits"].astype(str)
        starts = payload["start_frames"].astype(np.int64)
        crop_boxes = payload["crop_boxes"].astype(np.int64)
        person_found = payload["person_found"].astype(np.bool_)
        accessory_union = payload["accessory_union"].astype(np.bool_)
        frames_per_clip = int(np.asarray(payload["frames_per_clip"]).item())
        frame_stride = int(np.asarray(payload["frame_stride"]).item())
    row_count = features.shape[0]
    if features.shape != (row_count, CLIPS_PER_VIDEO, EMBEDDING_DIMENSION):
        raise RuntimeError(f"Unexpected crop feature shape: {features.shape}")
    expected_matrix_shape = (row_count, CLIPS_PER_VIDEO)
    if (
        labels.shape != (row_count,)
        or paths.shape != (row_count,)
        or splits.shape != (row_count,)
        or starts.shape != expected_matrix_shape
        or crop_boxes.shape != (*expected_matrix_shape, 4)
        or person_found.shape != expected_matrix_shape
        or accessory_union.shape != expected_matrix_shape
    ):
        raise RuntimeError("Crop cache arrays have inconsistent shapes")
    if frames_per_clip != FRAMES_PER_CLIP or frame_stride != FRAME_STRIDE:
        raise RuntimeError("Crop cache temporal settings are incompatible")
    if not np.isfinite(features).all():
        raise RuntimeError("Crop cache contains non-finite features")
    records: dict[str, dict[str, Any]] = {}
    for index, relative_path in enumerate(paths):
        key = str(relative_path).replace("\\", "/").casefold()
        if key in records:
            raise RuntimeError(f"Duplicate crop cache path: {relative_path}")
        records[key] = {
            "relative_path": str(relative_path).replace("\\", "/"),
            "label": int(labels[index]),
            "split": str(splits[index]),
            "features": features[index],
            "start_frames": starts[index],
            "crop_boxes": crop_boxes[index],
            "person_found": person_found[index],
            "accessory_union": accessory_union[index],
        }
    return records


def fallback_statistics(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    total_clips = len(records) * CLIPS_PER_VIDEO
    person_clips = sum(int(np.asarray(item["person_found"]).sum()) for item in records)
    union_clips = sum(
        int(np.asarray(item["accessory_union"]).sum()) for item in records
    )
    split_stats: dict[str, Any] = {}
    for split in sorted({str(item["split"]) for item in records}):
        subset = [item for item in records if item["split"] == split]
        split_total = len(subset) * CLIPS_PER_VIDEO
        split_person = sum(
            int(np.asarray(item["person_found"]).sum()) for item in subset
        )
        split_union = sum(
            int(np.asarray(item["accessory_union"]).sum()) for item in subset
        )
        split_stats[split] = {
            "videos": len(subset),
            "total_clips": split_total,
            "person_detected_clips": split_person,
            "fallback_full_frame_clips": split_total - split_person,
            "fallback_rate": (
                (split_total - split_person) / split_total if split_total else 0.0
            ),
            "accessory_union_clips": split_union,
        }
    return {
        "total_clips": total_clips,
        "person_detected_clips": person_clips,
        "fallback_full_frame_clips": total_clips - person_clips,
        "fallback_rate": (
            (total_clips - person_clips) / total_clips if total_clips else 0.0
        ),
        "accessory_union_clips": union_clips,
        "by_split": split_stats,
    }


def write_crop_cache(
    cache_path: Path,
    records: dict[str, dict[str, Any]],
    yolo_model_path: Path,
    yolo_sha256: str,
    manifest_sha256: str,
    confidence: float,
    image_size: int,
    extraction_seconds: float,
) -> None:
    ordered = sorted(
        records.values(),
        key=lambda item: baseline.natural_video_key(item["relative_path"]),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        features=np.stack([item["features"] for item in ordered]).astype(np.float32),
        labels=np.asarray([item["label"] for item in ordered], dtype=np.int64),
        paths=np.asarray([item["relative_path"] for item in ordered]),
        splits=np.asarray([item["split"] for item in ordered]),
        start_frames=np.stack([item["start_frames"] for item in ordered]).astype(
            np.int64
        ),
        crop_boxes=np.stack([item["crop_boxes"] for item in ordered]).astype(
            np.int64
        ),
        person_found=np.stack([item["person_found"] for item in ordered]).astype(
            np.bool_
        ),
        accessory_union=np.stack(
            [item["accessory_union"] for item in ordered]
        ).astype(np.bool_),
        yolo_sha256=np.asarray(yolo_sha256),
        yolo_confidence=np.asarray(confidence, dtype=np.float64),
        yolo_image_size=np.asarray(image_size, dtype=np.int64),
        crop_expansion=np.asarray(CROP_EXPANSION, dtype=np.float64),
        frames_per_clip=np.asarray(FRAMES_PER_CLIP, dtype=np.int64),
        frame_stride=np.asarray(FRAME_STRIDE, dtype=np.int64),
    )
    temporary.replace(cache_path)
    baseline.write_json(
        cache_path.with_suffix(".json"),
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": mvit.utc_now(),
            "feature_extractor": "TorchVision MViT-V2-S Kinetics-400",
            "view": "YOLO-guided person tube",
            "videos": len(ordered),
            "clips_per_video": CLIPS_PER_VIDEO,
            "edge_inclusive": True,
            "frames_per_clip": FRAMES_PER_CLIP,
            "frame_stride": FRAME_STRIDE,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "yolo_model": str(yolo_model_path),
            "yolo_sha256": yolo_sha256,
            "yolo_classes": {
                "person": PERSON_CLASS,
                **{name: class_id for class_id, name in ACCESSORY_NAMES.items()},
            },
            "yolo_confidence": confidence,
            "yolo_image_size": image_size,
            "person_selection": "largest visible person by clipped box area",
            "accessory_proximity": (
                "accessory center inside 1.35x person region or edge gap <= "
                "0.25x max person dimension"
            ),
            "crop_expansion": CROP_EXPANSION,
            "fallback": "full frame when no person is detected",
            "manifest_sha256": manifest_sha256,
            "last_extraction_seconds": extraction_seconds,
            "crop_fallback_statistics": fallback_statistics(ordered),
        },
    )


def extract_crop_embeddings(
    rows: Sequence[baseline.ManifestRow],
    args: argparse.Namespace,
    device: torch.device,
    yolo_model_path: Path,
    yolo_sha256: str,
) -> dict[str, dict[str, Any]]:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for crop extraction") from exc
    yolo_model = YOLO(str(yolo_model_path))
    backbone, preprocess = mvit.load_mvit_backbone(
        device,
        resolve_path(args.weights_file) if args.weights_file else None,
    )
    data_root = resolve_path(args.data_root)
    results: dict[str, dict[str, Any]] = {}
    pending: list[torch.Tensor] = []
    pending_locations: list[tuple[str, int]] = []
    feature_arrays = {
        row.relative_path: np.zeros(
            (CLIPS_PER_VIDEO, EMBEDDING_DIMENSION), dtype=np.float32
        )
        for row in rows
    }

    def flush() -> None:
        if not pending:
            return
        batch = torch.stack(pending).to(
            device, non_blocking=device.type == "cuda"
        )
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            embeddings = backbone(batch).float().cpu().numpy()
        if embeddings.shape[1:] != (EMBEDDING_DIMENSION,):
            raise RuntimeError(
                f"MViT produced unexpected crop embedding shape: {embeddings.shape}"
            )
        for embedding, (relative_path, clip_index) in zip(
            embeddings, pending_locations
        ):
            feature_arrays[relative_path][clip_index] = embedding
        pending.clear()
        pending_locations.clear()

    for video_index, row in enumerate(rows):
        video_path = data_root / Path(row.relative_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
        starts = mvit.edge_inclusive_starts(frame_count, CLIPS_PER_VIDEO)
        boxes, found, unioned, _ = detect_video_crops(
            video_path,
            starts,
            yolo_model,
            device,
            args.yolo_confidence,
            args.yolo_image_size,
        )
        for clip_index, tensor in iter_person_tube_clips(
            video_path, starts, boxes, preprocess
        ):
            pending.append(tensor)
            pending_locations.append((row.relative_path, clip_index))
            if len(pending) >= args.embedding_batch_size:
                flush()
        flush()
        features = feature_arrays[row.relative_path]
        if not np.isfinite(features).all() or np.any(
            np.linalg.norm(features, axis=-1) <= 0
        ):
            raise RuntimeError(f"Invalid crop embeddings for {row.relative_path}")
        results[row.relative_path.casefold()] = {
            "relative_path": row.relative_path,
            "label": row.label,
            "split": row.split,
            "features": features,
            "start_frames": starts,
            "crop_boxes": boxes,
            "person_found": found,
            "accessory_union": unioned,
        }
        print(
            f"YOLO-MViT crops {video_index + 1}/{len(rows)}: "
            f"{row.relative_path} | fallback={int((~found).sum())}/"
            f"{CLIPS_PER_VIDEO} | bag_union={int(unioned.sum())}",
            flush=True,
        )
    flush()
    _ = yolo_sha256
    return results


def ensure_crop_embeddings(
    args: argparse.Namespace,
    rows: Sequence[baseline.ManifestRow],
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Path,
    bool,
    str,
    dict[str, Any],
]:
    cache_path = resolve_path(args.crop_cache)
    yolo_model_path = resolve_path(args.yolo_model)
    if not yolo_model_path.is_file():
        raise FileNotFoundError(f"YOLO26s model is missing: {yolo_model_path}")
    yolo_sha256 = baseline.sha256_file(yolo_model_path)
    records = (
        {}
        if args.force_crop_extract
        else load_crop_cache(
            cache_path,
            yolo_sha256,
            args.yolo_confidence,
            args.yolo_image_size,
        )
    )
    requested = {row.relative_path.casefold(): row for row in rows}
    for key, record in records.items():
        manifest_row = requested.get(key)
        if manifest_row is None:
            # Preserve an explicitly opened test cache while default commands
            # remain train/validation-only.
            continue
        if (
            record["label"] != manifest_row.label
            or record["split"] != manifest_row.split
        ):
            raise RuntimeError(
                f"Crop cache label/split mismatch: {record['relative_path']}"
            )
    missing_rows = [
        row for row in rows if row.relative_path.casefold() not in records
    ]
    cache_reused = bool(records) and not missing_rows
    if missing_rows:
        started = time.perf_counter()
        records.update(
            extract_crop_embeddings(
                missing_rows, args, device, yolo_model_path, yolo_sha256
            )
        )
        write_crop_cache(
            cache_path=cache_path,
            records=records,
            yolo_model_path=yolo_model_path,
            yolo_sha256=yolo_sha256,
            manifest_sha256=baseline.sha256_file(
                resolve_path(args.split_manifest)
            ),
            confidence=args.yolo_confidence,
            image_size=args.yolo_image_size,
            extraction_seconds=time.perf_counter() - started,
        )
    aligned = [records[row.relative_path.casefold()] for row in rows]
    return (
        np.stack([item["features"] for item in aligned]).astype(np.float32),
        np.stack([item["start_frames"] for item in aligned]).astype(np.int64),
        np.stack([item["crop_boxes"] for item in aligned]).astype(np.int64),
        np.stack([item["person_found"] for item in aligned]).astype(np.bool_),
        np.stack([item["accessory_union"] for item in aligned]).astype(np.bool_),
        cache_path,
        cache_reused,
        yolo_sha256,
        fallback_statistics(aligned),
    )


def ensure_full_embeddings(
    args: argparse.Namespace,
    rows: Sequence[baseline.ManifestRow],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, bool]:
    full_args = SimpleNamespace(
        clips_per_video=CLIPS_PER_VIDEO,
        cache=args.full_cache,
        force_extract=args.force_full_extract,
        data_root=args.data_root,
        split_manifest=args.split_manifest,
        weights_file=args.weights_file,
        embedding_batch_size=args.embedding_batch_size,
    )
    return mvit.ensure_embeddings(full_args, rows, device)


def split_positions(
    rows: Sequence[baseline.ManifestRow],
) -> dict[str, np.ndarray]:
    return {
        split: np.asarray(
            [index for index, row in enumerate(rows) if row.split == split],
            dtype=np.int64,
        )
        for split in ("train", "val", "test")
    }


def validation_score(metrics: dict[str, Any]) -> float:
    return mvit.validation_score(metrics)


def train_seed(
    train_full: torch.Tensor,
    train_crop: torch.Tensor,
    train_labels: torch.Tensor,
    val_full: torch.Tensor,
    val_crop: torch.Tensor,
    val_labels: np.ndarray,
    *,
    view: str,
    pooling: str,
    fusion: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, float]:
    baseline.seed_everything(seed)
    model = DualViewMILHead(
        view=view,
        pooling=pooling,
        fusion=fusion,
        dropout=args.dropout,
        logsumexp_temperature=args.logsumexp_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    negative_count = int((train_labels == 0).sum().item())
    positive_count = int((train_labels == 1).sum().item())
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [negative_count / positive_count],
            dtype=torch.float32,
            device=device,
        )
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_score = -math.inf
    stale = 0
    for epoch in range(args.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        noisy_full = train_full + torch.randn_like(train_full) * args.noise_std
        noisy_crop = train_crop + torch.randn_like(train_crop) * args.noise_std
        loss = loss_function(
            model(noisy_full, noisy_crop), train_labels.float()
        )
        loss.backward()
        optimizer.step()
        if epoch % args.eval_every:
            continue
        model.eval()
        with torch.inference_mode():
            probabilities = torch.sigmoid(
                model(val_full, val_crop)
            ).cpu().numpy()
        metrics = baseline.binary_metrics(
            val_labels, probabilities, FIXED_THRESHOLD
        )
        score = validation_score(metrics)
        if score > best_score + 1e-8:
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_epoch = epoch
            best_score = score
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError(
            f"No checkpoint for view={view}, fusion={fusion}, "
            f"pooling={pooling}, seed={seed}"
        )
    return best_state, best_epoch, best_score


def ensemble_probabilities(
    states: Sequence[dict[str, Any]],
    checkpoint_like: Any,
    full_features: np.ndarray,
    crop_features: np.ndarray,
    full_scale: float,
    crop_scale: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    full_tensor = torch.from_numpy(
        full_features.astype(np.float32) / max(full_scale, 1e-6)
    ).to(device)
    crop_tensor = torch.from_numpy(
        crop_features.astype(np.float32) / max(crop_scale, 1e-6)
    ).to(device)

    def setting(name: str) -> Any:
        return (
            getattr(checkpoint_like, name)
            if hasattr(checkpoint_like, name)
            else checkpoint_like[name]
        )

    video_probabilities: list[np.ndarray] = []
    clip_probabilities: list[np.ndarray] = []
    for item in states:
        model = DualViewMILHead(
            view=str(setting("selected_view")),
            pooling=str(setting("selected_pooling")),
            fusion=str(setting("selected_fusion")),
            dropout=float(setting("dropout")),
            logsumexp_temperature=float(setting("logsumexp_temperature")),
        ).to(device)
        model.load_state_dict(item["state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            video_probabilities.append(
                torch.sigmoid(model(full_tensor, crop_tensor)).cpu().numpy()
            )
            clip_probabilities.append(
                torch.sigmoid(
                    model.clip_logits(full_tensor, crop_tensor)
                ).cpu().numpy()
            )
    return (
        np.mean(video_probabilities, axis=0),
        np.mean(clip_probabilities, axis=0),
    )


def candidate_settings(args: argparse.Namespace) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for pooling in args.poolings:
        candidates.append({"view": "full", "fusion": "none", "pooling": pooling})
        candidates.append({"view": "crop", "fusion": "none", "pooling": pooling})
        for fusion in args.dual_fusions:
            candidates.append(
                {"view": "dual", "fusion": fusion, "pooling": pooling}
            )
    return candidates


def build_predictions(
    rows: Sequence[baseline.ManifestRow],
    full_features: np.ndarray,
    crop_features: np.ndarray,
    starts: np.ndarray,
    crop_boxes: np.ndarray,
    person_found: np.ndarray,
    accessory_union: np.ndarray,
    checkpoint: dict[str, Any],
    included_splits: Sequence[str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positions = split_positions(rows)
    predictions: list[dict[str, Any]] = []
    metrics_by_split: dict[str, Any] = {}
    for split in included_splits:
        index = positions[split]
        probabilities, clip_probabilities = ensemble_probabilities(
            checkpoint["seed_checkpoints"],
            checkpoint,
            full_features[index],
            crop_features[index],
            float(checkpoint["full_feature_scale"]),
            float(checkpoint["crop_feature_scale"]),
            device,
        )
        labels = np.asarray(
            [rows[int(item)].label for item in index], dtype=np.int64
        )
        metrics_by_split[split] = baseline.binary_metrics(
            labels, probabilities, FIXED_THRESHOLD
        )
        for local_index, row_index in enumerate(index):
            row = rows[int(row_index)]
            probability = float(probabilities[local_index])
            predicted_label = int(probability >= FIXED_THRESHOLD)
            predictions.append(
                {
                    "relative_path": row.relative_path,
                    "split": split,
                    "actor_group": row.actor_group,
                    "ground_truth": row.label,
                    "ground_truth_name": baseline.LABEL_NAMES[row.label],
                    "shoplifting_probability": probability,
                    "predicted_label": predicted_label,
                    "predicted_name": baseline.LABEL_NAMES[predicted_label],
                    "correct": predicted_label == row.label,
                    "threshold": FIXED_THRESHOLD,
                    "selected_view": checkpoint["selected_view"],
                    "selected_fusion": checkpoint["selected_fusion"],
                    "selected_pooling": checkpoint["selected_pooling"],
                    "clip_start_frames": starts[int(row_index)].tolist(),
                    "crop_boxes_xyxy": crop_boxes[int(row_index)].tolist(),
                    "person_found": person_found[int(row_index)].tolist(),
                    "accessory_union": accessory_union[int(row_index)].tolist(),
                    "clip_shoplifting_probabilities": clip_probabilities[
                        local_index
                    ].tolist(),
                }
            )
    return predictions, metrics_by_split


def write_predictions(
    output_dir: Path,
    predictions: Sequence[dict[str, Any]],
    included_splits: Sequence[str],
) -> None:
    baseline.write_json(
        output_dir / "predictions.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": mvit.utc_now(),
            "included_splits": list(included_splits),
            "test_accessed": "test" in included_splits,
            "fixed_threshold": FIXED_THRESHOLD,
            "predictions": list(predictions),
        },
    )


def prepare_views(
    args: argparse.Namespace,
    rows: Sequence[baseline.ManifestRow],
    device: torch.device,
) -> dict[str, Any]:
    full_features, labels, full_starts, full_cache, full_reused = (
        ensure_full_embeddings(args, rows, device)
    )
    (
        crop_features,
        crop_starts,
        crop_boxes,
        person_found,
        accessory_union,
        crop_cache,
        crop_reused,
        yolo_sha256,
        crop_stats,
    ) = ensure_crop_embeddings(args, rows, device)
    if not np.array_equal(full_starts, crop_starts):
        raise RuntimeError("Full-frame and crop caches use different clip starts")
    expected_labels = np.asarray([row.label for row in rows], dtype=np.int64)
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError("Full-frame cache labels do not match the manifest")
    return {
        "full_features": full_features,
        "crop_features": crop_features,
        "labels": labels,
        "starts": crop_starts,
        "crop_boxes": crop_boxes,
        "person_found": person_found,
        "accessory_union": accessory_union,
        "full_cache": full_cache,
        "crop_cache": crop_cache,
        "full_cache_reused": full_reused,
        "crop_cache_reused": crop_reused,
        "yolo_sha256": yolo_sha256,
        "crop_fallback_statistics": crop_stats,
    }


def run_experiment(args: argparse.Namespace) -> None:
    device = baseline.choose_device(args.device)
    data_root = resolve_path(args.data_root)
    manifest_path = resolve_path(args.split_manifest)
    all_rows = baseline.load_manifest(manifest_path, data_root)
    rows = [row for row in all_rows if row.split in {"train", "val"}]
    prepared = prepare_views(args, rows, device)
    positions = split_positions(rows)
    train_index = positions["train"]
    val_index = positions["val"]
    full_scale = float(
        np.linalg.norm(
            prepared["full_features"][train_index], axis=-1, keepdims=True
        ).mean()
    )
    crop_scale = float(
        np.linalg.norm(
            prepared["crop_features"][train_index], axis=-1, keepdims=True
        ).mean()
    )
    if min(full_scale, crop_scale) <= 0 or not all(
        math.isfinite(value) for value in (full_scale, crop_scale)
    ):
        raise RuntimeError("Invalid train-only feature scales")
    full_tensor = torch.from_numpy(
        prepared["full_features"] / full_scale
    ).to(device)
    crop_tensor = torch.from_numpy(
        prepared["crop_features"] / crop_scale
    ).to(device)
    label_tensor = torch.from_numpy(prepared["labels"]).to(device)
    train_tensor_index = torch.from_numpy(train_index).to(device)
    val_tensor_index = torch.from_numpy(val_index).to(device)
    train_full = full_tensor.index_select(0, train_tensor_index)
    train_crop = crop_tensor.index_select(0, train_tensor_index)
    train_labels = label_tensor.index_select(0, train_tensor_index)
    val_full = full_tensor.index_select(0, val_tensor_index)
    val_crop = crop_tensor.index_select(0, val_tensor_index)
    val_labels = prepared["labels"][val_index]

    candidates: list[dict[str, Any]] = []
    for settings in candidate_settings(args):
        print(
            "Training candidate "
            f"view={settings['view']} fusion={settings['fusion']} "
            f"pooling={settings['pooling']}",
            flush=True,
        )
        seed_checkpoints: list[dict[str, Any]] = []
        for seed in args.seeds:
            state, epoch, score = train_seed(
                train_full,
                train_crop,
                train_labels,
                val_full,
                val_crop,
                val_labels,
                view=settings["view"],
                pooling=settings["pooling"],
                fusion=settings["fusion"],
                seed=seed,
                args=args,
                device=device,
            )
            seed_checkpoints.append(
                {
                    "seed": seed,
                    "best_epoch": epoch,
                    "best_validation_score": score,
                    "state_dict": state,
                }
            )
        candidate_descriptor = SimpleNamespace(
            selected_view=settings["view"],
            selected_pooling=settings["pooling"],
            selected_fusion=settings["fusion"],
            dropout=args.dropout,
            logsumexp_temperature=args.logsumexp_temperature,
        )
        probabilities, _ = ensemble_probabilities(
            seed_checkpoints,
            candidate_descriptor,
            prepared["full_features"][val_index],
            prepared["crop_features"][val_index],
            full_scale,
            crop_scale,
            device,
        )
        metrics = baseline.binary_metrics(
            val_labels, probabilities, FIXED_THRESHOLD
        )
        candidates.append(
            {
                **settings,
                "seed_checkpoints": seed_checkpoints,
                "validation_metrics": metrics,
                "validation_selection_score": validation_score(metrics),
            }
        )
    candidate_order = {
        (item["view"], item["fusion"], item["pooling"]): -index
        for index, item in enumerate(candidate_settings(args))
    }
    selected = max(
        candidates,
        key=lambda item: (
            item["validation_selection_score"],
            float(item["validation_metrics"]["balanced_accuracy"]),
            float(item["validation_metrics"]["roc_auc"]),
            candidate_order[
                (item["view"], item["fusion"], item["pooling"])
            ],
        ),
    )
    checkpoint = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "architecture": "YOLO-guided dual-view frozen MViT MIL",
        "manifest_sha256": baseline.sha256_file(manifest_path),
        "yolo_model_sha256": prepared["yolo_sha256"],
        "test_accessed_during_training": False,
        "selected_view": selected["view"],
        "selected_fusion": selected["fusion"],
        "selected_pooling": selected["pooling"],
        "clips_per_video": CLIPS_PER_VIDEO,
        "frames_per_clip": FRAMES_PER_CLIP,
        "frame_stride": FRAME_STRIDE,
        "full_feature_scale": full_scale,
        "crop_feature_scale": crop_scale,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "dropout": args.dropout,
        "logsumexp_temperature": args.logsumexp_temperature,
        "threshold": FIXED_THRESHOLD,
        "seed_checkpoints": selected["seed_checkpoints"],
    }
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    temporary = checkpoint_path.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    predictions, metrics = build_predictions(
        rows,
        prepared["full_features"],
        prepared["crop_features"],
        prepared["starts"],
        prepared["crop_boxes"],
        prepared["person_found"],
        prepared["accessory_union"],
        checkpoint,
        ("train", "val"),
        device,
    )
    write_predictions(output_dir, predictions, ("train", "val"))
    summary_candidates = [
        {
            "view": item["view"],
            "fusion": item["fusion"],
            "pooling": item["pooling"],
            "validation_selection_score": item[
                "validation_selection_score"
            ],
            "validation_metrics": item["validation_metrics"],
            "seeds": [
                {
                    "seed": state["seed"],
                    "best_epoch": state["best_epoch"],
                    "best_validation_score": state[
                        "best_validation_score"
                    ],
                }
                for state in item["seed_checkpoints"]
            ],
        }
        for item in candidates
    ]
    baseline.write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": mvit.utc_now(),
            "command": "run",
            "test_accessed": False,
            "selection_scope": "validation split only",
            "feature_extractor": "TorchVision MViT-V2-S Kinetics-400",
            "yolo_participation": (
                "YOLO26s defines label-free person-tube crops used by crop "
                "and dual-view candidates"
            ),
            "yolo_model": str(resolve_path(args.yolo_model)),
            "yolo_sha256": prepared["yolo_sha256"],
            "clips_per_video": CLIPS_PER_VIDEO,
            "frames_per_clip": FRAMES_PER_CLIP,
            "frame_stride": FRAME_STRIDE,
            "crop_expansion": CROP_EXPANSION,
            "selected_view": selected["view"],
            "selected_fusion": selected["fusion"],
            "selected_pooling": selected["pooling"],
            "full_feature_scale_source": "train split only",
            "crop_feature_scale_source": "train split only",
            "full_feature_scale": full_scale,
            "crop_feature_scale": crop_scale,
            "fixed_threshold": FIXED_THRESHOLD,
            "candidate_results": summary_candidates,
            "metrics": metrics,
            "crop_fallback_statistics": prepared[
                "crop_fallback_statistics"
            ],
            "caches": {
                "full": str(prepared["full_cache"]),
                "crop": str(prepared["crop_cache"]),
                "full_reused": prepared["full_cache_reused"],
                "crop_reused": prepared["crop_cache_reused"],
            },
            "outputs": {
                "checkpoint": str(checkpoint_path),
                "predictions": str(output_dir / "predictions.json"),
            },
        },
    )
    print(
        json.dumps(
            {
                "selected_view": selected["view"],
                "selected_fusion": selected["fusion"],
                "selected_pooling": selected["pooling"],
                "train": metrics["train"],
                "val": metrics["val"],
                "crop_fallback_statistics": prepared[
                    "crop_fallback_statistics"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "seed_checkpoints" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint: {path}")
    if len(checkpoint["seed_checkpoints"]) != 5:
        raise RuntimeError("Checkpoint must contain five seed heads")
    if int(checkpoint["clips_per_video"]) != CLIPS_PER_VIDEO:
        raise RuntimeError("Checkpoint must use 21 clips")
    if float(checkpoint["threshold"]) != FIXED_THRESHOLD:
        raise RuntimeError("Checkpoint threshold must be 0.50")
    return checkpoint


def evaluate_experiment(args: argparse.Namespace) -> None:
    device = baseline.choose_device(args.device)
    data_root = resolve_path(args.data_root)
    manifest_path = resolve_path(args.split_manifest)
    output_dir = resolve_path(args.output_dir)
    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint
        else output_dir / "checkpoint.pt"
    )
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["manifest_sha256"] != baseline.sha256_file(manifest_path):
        raise RuntimeError("Checkpoint manifest hash mismatch")
    current_yolo_sha256 = baseline.sha256_file(resolve_path(args.yolo_model))
    if checkpoint["yolo_model_sha256"] != current_yolo_sha256:
        raise RuntimeError("Checkpoint YOLO model hash mismatch")
    included_splits = (
        ("train", "val", "test")
        if args.include_test
        else ("train", "val")
    )
    all_rows = baseline.load_manifest(manifest_path, data_root)
    rows = [row for row in all_rows if row.split in included_splits]
    prepared = prepare_views(args, rows, device)
    predictions, metrics = build_predictions(
        rows,
        prepared["full_features"],
        prepared["crop_features"],
        prepared["starts"],
        prepared["crop_boxes"],
        prepared["person_found"],
        prepared["accessory_union"],
        checkpoint,
        included_splits,
        device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir, predictions, included_splits)
    baseline.write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": mvit.utc_now(),
            "command": "evaluate",
            "test_accessed": args.include_test,
            "included_splits": list(included_splits),
            "selection_scope": "validation split only during training",
            "selected_view": checkpoint["selected_view"],
            "selected_fusion": checkpoint["selected_fusion"],
            "selected_pooling": checkpoint["selected_pooling"],
            "fixed_threshold": FIXED_THRESHOLD,
            "yolo_model": str(resolve_path(args.yolo_model)),
            "yolo_sha256": current_yolo_sha256,
            "metrics": metrics,
            "crop_fallback_statistics": prepared[
                "crop_fallback_statistics"
            ],
            "caches": {
                "full": str(prepared["full_cache"]),
                "crop": str(prepared["crop_cache"]),
                "full_reused": prepared["full_cache_reused"],
                "crop_reused": prepared["crop_cache_reused"],
            },
            "checkpoint": str(checkpoint_path),
        },
    )
    print(json.dumps(metrics, indent=2), flush=True)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--full-cache", type=Path, default=DEFAULT_FULL_CACHE)
    parser.add_argument("--crop-cache", type=Path, default=DEFAULT_CROP_CACHE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights-file", type=Path, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=2)
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-image-size", type=int, default=640)
    parser.add_argument(
        "--force-full-extract",
        action="store_true",
        help="Rebuild included rows in the full-frame MViT cache.",
    )
    parser.add_argument(
        "--force-crop-extract",
        action="store_true",
        help="Rebuild included rows in the YOLO-guided crop cache.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "YOLO26s-guided person-tube plus full-frame MViT MIL experiment."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run",
        help="Prepare train+val views, compare candidates, and select on val.",
    )
    add_common_arguments(run_parser)
    run_parser.add_argument(
        "--poolings",
        nargs="+",
        choices=POOLING_CHOICES,
        default=list(POOLING_CHOICES),
    )
    run_parser.add_argument(
        "--dual-fusions",
        nargs="+",
        choices=DUAL_FUSION_CHOICES,
        default=list(DUAL_FUSION_CHOICES),
    )
    run_parser.add_argument(
        "--seeds", type=int, nargs=5, default=list(DEFAULT_SEEDS)
    )
    run_parser.add_argument("--max-epochs", type=int, default=1200)
    run_parser.add_argument("--eval-every", type=int, default=5)
    run_parser.add_argument("--patience", type=int, default=80)
    run_parser.add_argument("--learning-rate", type=float, default=0.0015)
    run_parser.add_argument("--weight-decay", type=float, default=0.02)
    run_parser.add_argument("--noise-std", type=float, default=0.002)
    run_parser.add_argument("--dropout", type=float, default=0.30)
    run_parser.add_argument(
        "--logsumexp-temperature", type=float, default=1.0
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate train+val; --include-test explicitly opens test.",
    )
    add_common_arguments(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, default=None)
    evaluate_parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly extract/read and score the fixed internal test split.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive")
    if not 0 < args.yolo_confidence <= 1:
        raise ValueError("--yolo-confidence must be in (0, 1]")
    if args.yolo_image_size <= 0:
        raise ValueError("--yolo-image-size must be positive")
    if args.command == "run":
        if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
            raise ValueError("--seeds must contain exactly five distinct values")
        if len(set(args.poolings)) != len(args.poolings):
            raise ValueError("--poolings contains duplicates")
        if len(set(args.dual_fusions)) != len(args.dual_fusions):
            raise ValueError("--dual-fusions contains duplicates")
        if min(args.max_epochs, args.eval_every, args.patience) <= 0:
            raise ValueError("epoch and patience settings must be positive")
        if args.logsumexp_temperature <= 0:
            raise ValueError("--logsumexp-temperature must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.command == "run":
        run_experiment(args)
    elif args.command == "evaluate":
        evaluate_experiment(args)
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
