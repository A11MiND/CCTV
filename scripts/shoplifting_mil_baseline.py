"""Weakly supervised R3D-18 MIL baseline for the Shoplifting Video Dataset.

The source dataset provides one Normal/Shoplifting label per video.  This
script therefore treats each video as a bag of seven deterministic temporal
clips.  A frozen Kinetics-400 R3D-18 produces one 512-dimensional embedding per
clip and a small MLP produces clip logits.  The video logit is the mean of the
two largest clip logits:

    clip_logits = MLP(video_clip_embeddings)
    video_logit = mean(top_k(clip_logits, k=2))
    shoplifting_probability = sigmoid(video_logit)

All final classifications use a fixed threshold of 0.50.  Checkpoint selection
uses validation ROC-AUC and validation balanced accuracy only.  Test, OOD, and
qualitative examples never participate in optimization or model selection.

Default inputs:

    data/shoplifting-video-dataset/
    docs/results/shoplifting/actor-clothing-disjoint-split.csv

Primary outputs:

    output/shoplifting-mil-baseline/
        dataset-audit.json / .csv
        config.json / .csv
        checkpoint.pt / checkpoint.json
        history.json / .csv
        metrics.json / .csv
        predictions.json / .csv
        confusion-matrices.csv
        metrics-dashboard.png
        confusion-matrices.png
        demo-summary.json
        demo-windows.csv

    public/assets/video/shoplifting-mil-heldout-demo.mp4

Examples, run from the repository root:

    .venv-yolo\\Scripts\\python.exe scripts\\shoplifting_mil_baseline.py run

    .venv-yolo\\Scripts\\python.exe scripts\\shoplifting_mil_baseline.py evaluate

    .venv-yolo\\Scripts\\python.exe scripts\\shoplifting_mil_baseline.py demo

The embedding cache is reused when its paths, labels, sampling fractions, and
shape match the reviewed 182-video manifest.  Use ``run --force-extract`` to
replace it deliberately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "shoplifting-video-dataset"
DEFAULT_SPLIT_MANIFEST = (
    ROOT / "docs" / "results" / "shoplifting" / "actor-clothing-disjoint-split.csv"
)
DEFAULT_CACHE = ROOT / "data" / "cache" / "r3d18-video-embeddings.npz"
DEFAULT_CACHE_METADATA = ROOT / "data" / "cache" / "r3d18-video-embeddings.json"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "shoplifting-mil-baseline"
DEFAULT_DEMO_OUTPUT = (
    ROOT / "public" / "assets" / "video" / "shoplifting-mil-heldout-demo.mp4"
)
DEFAULT_YOLO_MODEL = ROOT / "models" / "yolo26s.pt"

EXPECTED_VIDEO_COUNT = 182
CLIPS_PER_VIDEO = 7
FRAMES_PER_CLIP = 16
FRAME_STRIDE = 4
EMBEDDING_DIMENSION = 512
HIDDEN_DIMENSION = 64
TOP_CLIPS = 2
FIXED_THRESHOLD = 0.50
DEFAULT_SEEDS = (11, 23, 37, 51, 79)
EVALUATION_SPLITS = ("train", "val", "test", "ood", "qualitative")
ALLOWED_SPLITS = (*EVALUATION_SPLITS, "excluded")
LABEL_NAMES = {0: "Normal", 1: "Shoplifting"}
SCRIPT_VERSION = "1.0.0"

if (ROOT / "data" / "torch-cache").is_dir():
    os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "torch-cache"))


@dataclass(frozen=True)
class ManifestRow:
    """One reviewed source-video assignment."""

    relative_path: str
    split: str
    actor_group: str
    reason: str
    label: int


@dataclass
class PreparedData:
    """Audited source records and aligned embedding tensors."""

    rows: list[ManifestRow]
    features: np.ndarray
    labels: np.ndarray
    fractions: np.ndarray
    split_indices: dict[str, np.ndarray]
    manifest_sha256: str
    audit_sha256: str
    cache_reused: bool


class MILHead(nn.Module):
    """512 -> 64 -> 1 clip head with top-2 mean video aggregation."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(EMBEDDING_DIMENSION, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIMENSION, 1),
        )

    def clip_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings).squeeze(-1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        clip_logits = self.clip_logits(embeddings)
        top_count = min(TOP_CLIPS, clip_logits.shape[1])
        return clip_logits.topk(top_count, dim=1).values.mean(dim=1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def json_ready(value: Any) -> Any:
    """Convert numpy, tensor, and non-finite values to strict JSON data."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_ready(row.get(key, "")) for key in fieldnames})
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def label_from_relative_path(relative_path: str) -> int:
    class_name = PurePosixPath(relative_path).parts[0].casefold()
    if class_name == "normal":
        return 0
    if class_name == "shoplifting":
        return 1
    raise ValueError(f"Unsupported class folder in {relative_path!r}")


def natural_video_key(relative_path: str) -> tuple[str, int, str]:
    path = PurePosixPath(relative_path)
    try:
        number = int(path.stem.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        number = sys.maxsize
    return path.parent.name.casefold(), number, relative_path.casefold()


def load_manifest(path: Path, data_root: Path) -> list[ManifestRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    required = {"relative_path", "split", "actor_group", "reason"}
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise RuntimeError(
                f"Split manifest is missing columns: {sorted(missing_columns)}"
            )
        for line_number, raw in enumerate(reader, start=2):
            relative_path = str(raw["relative_path"]).replace("\\", "/").strip()
            split = str(raw["split"]).casefold().strip()
            if not relative_path or PurePosixPath(relative_path).is_absolute():
                raise RuntimeError(
                    f"Invalid relative_path at manifest line {line_number}"
                )
            if ".." in PurePosixPath(relative_path).parts:
                raise RuntimeError(
                    f"Path traversal is not allowed at manifest line {line_number}"
                )
            if split not in ALLOWED_SPLITS:
                raise RuntimeError(
                    f"Unsupported split {split!r} at manifest line {line_number}"
                )
            key = relative_path.casefold()
            if key in seen:
                raise RuntimeError(f"Duplicate manifest path: {relative_path}")
            seen.add(key)
            rows.append(
                ManifestRow(
                    relative_path=relative_path,
                    split=split,
                    actor_group=str(raw["actor_group"]).strip(),
                    reason=str(raw["reason"]).strip(),
                    label=label_from_relative_path(relative_path),
                )
            )

    discovered = {
        item.relative_to(data_root).as_posix()
        for item in data_root.rglob("*.mp4")
        if item.is_file()
    }
    manifest_paths = {row.relative_path for row in rows}
    missing = sorted(discovered - manifest_paths, key=natural_video_key)
    extra = sorted(manifest_paths - discovered, key=natural_video_key)
    if missing or extra:
        raise RuntimeError(
            "Manifest/dataset mismatch: "
            f"unassigned_dataset_files={missing}, missing_manifest_files={extra}"
        )
    if len(rows) != EXPECTED_VIDEO_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_VIDEO_COUNT} manifest rows, found {len(rows)}"
        )
    return rows


def decode_fourcc(value: int) -> str:
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4)).strip()


def audit_video(row: ManifestRow, data_root: Path) -> dict[str, Any]:
    path = data_root / Path(row.relative_path)
    record: dict[str, Any] = {
        **asdict(row),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_seconds": 0.0,
        "codec": "",
        "valid": False,
        "error": "",
    }
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the video")
        record["width"] = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        record["height"] = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        record["fps"] = float(capture.get(cv2.CAP_PROP_FPS))
        record["frame_count"] = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        record["codec"] = decode_fourcc(int(capture.get(cv2.CAP_PROP_FOURCC)))
        if record["fps"] > 0:
            record["duration_seconds"] = record["frame_count"] / record["fps"]
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("OpenCV could not decode the first frame")
        if record["width"] <= 0 or record["height"] <= 0:
            raise RuntimeError("Invalid frame dimensions")
        if record["fps"] <= 0 or record["frame_count"] <= 0:
            raise RuntimeError("Invalid FPS or frame count")
        record["valid"] = True
    except Exception as exc:  # retain a complete 182-row audit before failing
        record["error"] = str(exc)
    finally:
        capture.release()
    return record


def audit_dataset(
    rows: Sequence[ManifestRow],
    data_root: Path,
    output_dir: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    print(f"Auditing all {len(rows)} source videos...", flush=True)
    records = [audit_video(row, data_root) for row in rows]
    invalid = [record for record in records if not record["valid"]]
    hashes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        hashes[str(record["sha256"])].append(str(record["relative_path"]))
    exact_duplicate_groups = [
        sorted(paths, key=natural_video_key)
        for paths in hashes.values()
        if len(paths) > 1
    ]
    split_counts: dict[str, dict[str, int]] = {}
    for split in ALLOWED_SPLITS:
        split_rows = [record for record in records if record["split"] == split]
        split_counts[split] = {
            "normal": sum(record["label"] == 0 for record in split_rows),
            "shoplifting": sum(record["label"] == 1 for record in split_rows),
            "total": len(split_rows),
        }
    audit_sha256 = hashlib.sha256(
        "".join(
            f"{record['relative_path']}\0{record['sha256']}\n" for record in records
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "dataset_root": str(data_root),
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "audit_sha256": audit_sha256,
        "video_count": len(records),
        "valid_video_count": len(records) - len(invalid),
        "invalid_video_count": len(invalid),
        "class_counts": {
            LABEL_NAMES[label]: sum(record["label"] == label for record in records)
            for label in LABEL_NAMES
        },
        "split_counts": split_counts,
        "exact_duplicate_groups": exact_duplicate_groups,
        "records": records,
    }
    write_json(output_dir / "dataset-audit.json", payload)
    audit_fields = (
        "relative_path",
        "split",
        "actor_group",
        "label",
        "size_bytes",
        "sha256",
        "width",
        "height",
        "fps",
        "frame_count",
        "duration_seconds",
        "codec",
        "valid",
        "error",
        "reason",
    )
    write_csv(output_dir / "dataset-audit.csv", records, audit_fields)
    if invalid:
        raise RuntimeError(
            "Dataset audit found invalid videos: "
            + ", ".join(str(record["relative_path"]) for record in invalid)
        )
    return records, audit_sha256


def canonical_cache_path(raw_path: str, data_root: Path) -> str:
    normalized = str(raw_path).replace("\\", "/")
    prefixes = (
        data_root.name.rstrip("/") + "/",
        "data/shoplifting-video-dataset/",
        "shoplifting-video-dataset/",
    )
    lowered = normalized.casefold()
    for prefix in prefixes:
        position = lowered.find(prefix.casefold())
        if position >= 0:
            return normalized[position + len(prefix) :]
    parts = PurePosixPath(normalized).parts
    for index, part in enumerate(parts):
        if part.casefold() in {"normal", "shoplifting"}:
            return PurePosixPath(*parts[index:]).as_posix()
    raise RuntimeError(f"Could not map cached path to the dataset: {raw_path!r}")


def load_validated_cache(
    cache_path: Path,
    rows: Sequence[ManifestRow],
    data_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as payload:
        required = {"features", "labels", "paths", "fractions"}
        missing = required - set(payload.files)
        if missing:
            raise RuntimeError(f"Embedding cache is missing arrays: {sorted(missing)}")
        features = payload["features"].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
        raw_paths = payload["paths"].astype(str)
        fractions = payload["fractions"].astype(np.float64)
    if features.shape != (
        EXPECTED_VIDEO_COUNT,
        CLIPS_PER_VIDEO,
        EMBEDDING_DIMENSION,
    ):
        raise RuntimeError(f"Unexpected embedding shape: {features.shape}")
    if labels.shape != (EXPECTED_VIDEO_COUNT,) or raw_paths.shape != (
        EXPECTED_VIDEO_COUNT,
    ):
        raise RuntimeError("Embedding labels/paths do not contain 182 rows")
    expected_fractions = np.linspace(0.12, 0.88, CLIPS_PER_VIDEO)
    if fractions.shape != expected_fractions.shape or not np.allclose(
        fractions, expected_fractions, atol=1e-8
    ):
        raise RuntimeError(
            "Embedding cache does not use the required seven center fractions"
        )
    if not np.isfinite(features).all() or np.any(
        np.linalg.norm(features, axis=-1) <= 0
    ):
        raise RuntimeError("Embedding cache contains non-finite or empty features")

    cache_index: dict[str, int] = {}
    for index, raw_path in enumerate(raw_paths):
        relative_path = canonical_cache_path(str(raw_path), data_root)
        key = relative_path.casefold()
        if key in cache_index:
            raise RuntimeError(f"Duplicate cached path: {relative_path}")
        cache_index[key] = index
    expected_keys = {row.relative_path.casefold() for row in rows}
    if set(cache_index) != expected_keys:
        raise RuntimeError("Embedding cache paths do not match the reviewed manifest")
    order = np.asarray(
        [cache_index[row.relative_path.casefold()] for row in rows], dtype=np.int64
    )
    aligned_features = features[order]
    aligned_labels = labels[order]
    expected_labels = np.asarray([row.label for row in rows], dtype=np.int64)
    if not np.array_equal(aligned_labels, expected_labels):
        raise RuntimeError("Embedding cache labels do not match manifest class folders")
    return aligned_features, aligned_labels, fractions


def load_r3d_backbone(
    device: torch.device,
    weights_file: Path | None,
) -> tuple[nn.Module, nn.Module]:
    weights = R3D_18_Weights.KINETICS400_V1
    if weights_file is None:
        try:
            model = r3d_18(weights=weights, progress=True)
        except Exception as exc:
            raise RuntimeError(
                "Could not load the official TorchVision R3D-18 Kinetics-400 "
                "weights. Populate data/torch-cache once or pass --weights-file."
            ) from exc
    else:
        model = r3d_18(weights=None)
        payload = torch.load(weights_file, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unsupported R3D-18 state dict: {weights_file}")
        state = {
            str(key).removeprefix("module."): value
            for key, value in payload.items()
            if isinstance(value, torch.Tensor)
        }
        model.load_state_dict(state, strict=True)
    model.fc = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval().to(device)
    return model, weights.transforms()


def sample_video_clips(
    path: Path,
    center_fractions: np.ndarray,
    preprocess: nn.Module,
) -> list[torch.Tensor]:
    """Decode the exact seven full-frame temporal clips used by the temp script."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Invalid frame count for {path}")
    span = (FRAMES_PER_CLIP - 1) * FRAME_STRIDE
    clip_indices: list[list[int]] = []
    for center_fraction in center_fractions:
        center = round(float(center_fraction) * max(0, frame_count - 1))
        start = max(0, min(frame_count - 1 - span, center - span // 2))
        clip_indices.append(
            [
                min(frame_count - 1, start + index * FRAME_STRIDE)
                for index in range(FRAMES_PER_CLIP)
            ]
        )
    needed = {frame_index for indices in clip_indices for frame_index in indices}
    decoded: dict[int, np.ndarray] = {}
    try:
        for frame_index in range(max(needed) + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read frame {frame_index} from {path}")
            if frame_index in needed:
                decoded[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    clips: list[torch.Tensor] = []
    for indices in clip_indices:
        tensor = (
            torch.from_numpy(np.stack([decoded[index] for index in indices]))
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        clips.append(preprocess(tensor))
    return clips


def extract_embeddings(
    rows: Sequence[ManifestRow],
    data_root: Path,
    cache_path: Path,
    cache_metadata_path: Path,
    device: torch.device,
    weights_file: Path | None,
    batch_size: int,
    manifest_path: Path,
    audit_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractions = np.linspace(0.12, 0.88, CLIPS_PER_VIDEO)
    model, preprocess = load_r3d_backbone(device, weights_file)
    features = np.zeros(
        (len(rows), CLIPS_PER_VIDEO, EMBEDDING_DIMENSION), dtype=np.float32
    )
    pending: list[torch.Tensor] = []
    locations: list[tuple[int, int]] = []
    started = time.perf_counter()

    def flush() -> None:
        if not pending:
            return
        batch = torch.stack(pending).to(device, non_blocking=device.type == "cuda")
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            batch_features = model(batch).float().cpu().numpy()
        for feature, (video_index, clip_index) in zip(batch_features, locations):
            features[video_index, clip_index] = feature
        pending.clear()
        locations.clear()

    for video_index, row in enumerate(rows):
        clips = sample_video_clips(
            data_root / Path(row.relative_path), fractions, preprocess
        )
        for clip_index, clip in enumerate(clips):
            pending.append(clip)
            locations.append((video_index, clip_index))
            if len(pending) >= batch_size:
                flush()
        if (video_index + 1) % 20 == 0:
            print(
                f"Extracted embeddings for {video_index + 1}/{len(rows)} videos",
                flush=True,
            )
    flush()
    if not np.isfinite(features).all():
        raise RuntimeError("Feature extraction produced non-finite values")
    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    cache_paths = np.asarray(
        [
            (data_root / Path(row.relative_path))
            .relative_to(ROOT)
            .as_posix()
            for row in rows
        ]
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        features=features,
        labels=labels,
        paths=cache_paths,
        fractions=fractions,
    )
    temporary.replace(cache_path)
    metadata = {
        "schema_version": 2,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "dataset": str(data_root.relative_to(ROOT)).replace("\\", "/"),
        "split_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "split_manifest_sha256": sha256_file(manifest_path),
        "source_audit_sha256": audit_sha256,
        "videos": len(rows),
        "normal": int((labels == 0).sum()),
        "shoplifting": int((labels == 1).sum()),
        "clips_per_video": CLIPS_PER_VIDEO,
        "frames_per_clip": FRAMES_PER_CLIP,
        "frame_stride": FRAME_STRIDE,
        "center_fractions": fractions.tolist(),
        "full_frame": True,
        "embedding": "TorchVision R3D-18 Kinetics-400 penultimate layer",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "device": str(device),
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_json(cache_metadata_path, metadata)
    return features, labels, fractions


def split_indices(rows: Sequence[ManifestRow]) -> dict[str, np.ndarray]:
    result = {
        split: np.asarray(
            [index for index, row in enumerate(rows) if row.split == split],
            dtype=np.int64,
        )
        for split in ALLOWED_SPLITS
    }
    for split in ("train", "val", "test", "ood", "qualitative"):
        if result[split].size == 0:
            raise RuntimeError(f"Reviewed manifest has no {split} rows")
    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    for split in ("train", "val", "test", "ood"):
        if np.unique(labels[result[split]]).size != 2:
            raise RuntimeError(f"{split} split must contain both classes")
    return result


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def prepare_data(args: argparse.Namespace, device: torch.device) -> PreparedData:
    data_root = resolve_path(args.data_root)
    manifest_path = resolve_path(args.split_manifest)
    cache_path = resolve_path(args.cache)
    cache_metadata_path = resolve_path(args.cache_metadata)
    output_dir = resolve_path(args.output_dir)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")
    rows = load_manifest(manifest_path, data_root)
    _, audit_sha256 = audit_dataset(rows, data_root, output_dir, manifest_path)
    cache_reused = False
    if cache_path.is_file() and not getattr(args, "force_extract", False):
        try:
            features, labels, fractions = load_validated_cache(
                cache_path, rows, data_root
            )
            cache_reused = True
            print(f"Reusing validated embedding cache: {cache_path}", flush=True)
        except Exception as exc:
            raise RuntimeError(
                f"Existing cache failed validation: {exc}. "
                "Use --force-extract to replace it."
            ) from exc
    else:
        features, labels, fractions = extract_embeddings(
            rows=rows,
            data_root=data_root,
            cache_path=cache_path,
            cache_metadata_path=cache_metadata_path,
            device=device,
            weights_file=(
                resolve_path(args.weights_file) if args.weights_file else None
            ),
            batch_size=args.embedding_batch_size,
            manifest_path=manifest_path,
            audit_sha256=audit_sha256,
        )
    return PreparedData(
        rows=rows,
        features=features,
        labels=labels,
        fractions=fractions,
        split_indices=split_indices(rows),
        manifest_sha256=sha256_file(manifest_path),
        audit_sha256=audit_sha256,
        cache_reused=cache_reused,
    )


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def roc_auc_binary(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return None
    wins = 0.0
    for score in positive:
        wins += float(np.count_nonzero(score > negative))
        wins += 0.5 * float(np.count_nonzero(score == negative))
    return wins / float(positive.size * negative.size)


def binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    true_negative = int(np.sum((truth == 0) & (predictions == 0)))
    false_positive = int(np.sum((truth == 0) & (predictions == 1)))
    false_negative = int(np.sum((truth == 1) & (predictions == 0)))
    true_positive = int(np.sum((truth == 1) & (predictions == 1)))
    accuracy = safe_divide(true_positive + true_negative, truth.size)
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    specificity = safe_divide(true_negative, true_negative + false_positive)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "unit": "source_video",
        "threshold": threshold,
        "support": int(truth.size),
        "normal_support": int(np.sum(truth == 0)),
        "shoplifting_support": int(np.sum(truth == 1)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc_binary(truth, scores),
        "confusion_matrix": {
            "labels": ["Normal", "Shoplifting"],
            "values": [
                [true_negative, false_positive],
                [false_negative, true_positive],
            ],
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
        },
    }


def validation_selection_score(metrics: dict[str, Any]) -> float:
    """Retain the temp experiment's BA + 0.05*AUC validation-only score."""

    auc = metrics["roc_auc"]
    if auc is None:
        return -math.inf
    return float(metrics["balanced_accuracy"]) + 0.05 * float(auc)


def train_one_seed(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    *,
    seed: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    noise_std: float,
    max_epochs: int,
    eval_every: int,
    patience: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, float, list[dict[str, Any]]]:
    seed_everything(seed)
    model = MILHead(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    negative_count = int((train_y == 0).sum().item())
    positive_count = int((train_y == 1).sum().item())
    if not negative_count or not positive_count:
        raise RuntimeError("Training split requires both classes")
    pos_weight = torch.tensor(
        [negative_count / positive_count], dtype=torch.float32, device=device
    )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    best_epoch = -1
    stale_evaluations = 0
    history: list[dict[str, Any]] = []

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        noisy = train_x + torch.randn_like(train_x) * noise_std
        train_logits = model(noisy)
        loss = loss_function(train_logits, train_y.float())
        loss.backward()
        optimizer.step()
        if epoch % eval_every:
            continue

        model.eval()
        with torch.inference_mode():
            val_probabilities = torch.sigmoid(model(val_x)).detach().cpu().numpy()
        val_metrics = binary_metrics(
            val_y, val_probabilities, threshold=FIXED_THRESHOLD
        )
        score = validation_selection_score(val_metrics)
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "validation_selection_score": score,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                "validation_roc_auc": val_metrics["roc_auc"],
                "selected_at_this_epoch": False,
            }
        )
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            history[-1]["selected_at_this_epoch"] = True
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        if stale_evaluations >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"Seed {seed} did not produce a valid checkpoint")
    return best_state, best_epoch, best_score, history


def environment_info(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def flatten_mapping(
    mapping: dict[str, Any], prefix: str = ""
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(flatten_mapping(value, full_key))
        else:
            rendered = json.dumps(json_ready(value), ensure_ascii=False)
            rows.append({"key": full_key, "value": rendered})
    return rows


def experiment_config(
    args: argparse.Namespace,
    prepared: PreparedData,
    device: torch.device,
    feature_scale: float,
) -> dict[str, Any]:
    config = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "dataset_root": str(resolve_path(args.data_root)),
        "split_manifest": str(resolve_path(args.split_manifest)),
        "split_manifest_sha256": prepared.manifest_sha256,
        "source_audit_sha256": prepared.audit_sha256,
        "embedding_cache": str(resolve_path(args.cache)),
        "embedding_cache_reused": prepared.cache_reused,
        "feature_extractor": "TorchVision R3D-18 Kinetics-400 penultimate layer",
        "full_frame_clips": True,
        "clips_per_video": CLIPS_PER_VIDEO,
        "frames_per_clip": FRAMES_PER_CLIP,
        "frame_stride": FRAME_STRIDE,
        "clip_center_fractions": prepared.fractions.tolist(),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "feature_scale": feature_scale,
        "head": "Linear(512,64) -> ReLU -> Dropout(0.25) -> Linear(64,1)",
        "video_logit": "mean(top_2(clip_logits))",
        "probability": "sigmoid(video_logit)",
        "ensemble": "mean of five seed video probabilities",
        "seeds": list(args.seeds),
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "noise_std": args.noise_std,
        "dropout": args.dropout,
        "max_epochs": args.max_epochs,
        "evaluation_interval_epochs": args.eval_every,
        "early_stopping_patience_evaluations": args.patience,
        "checkpoint_selection": (
            "validation balanced_accuracy + 0.05 * validation ROC-AUC"
        ),
        "selection_data": "validation split only",
        "reporting_threshold": FIXED_THRESHOLD,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_method": "class-stratified percentile interval",
        "reporting_unit": "source video",
        "environment": environment_info(device),
    }
    output_dir = resolve_path(args.output_dir)
    write_json(output_dir / "config.json", config)
    write_csv(
        output_dir / "config.csv",
        flatten_mapping(config),
        ("key", "value"),
    )
    return config


def train_pipeline(
    args: argparse.Namespace,
    prepared: PreparedData,
    device: torch.device,
) -> Path:
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        raise RuntimeError("--seeds must contain exactly five distinct integers")
    train_index = prepared.split_indices["train"]
    val_index = prepared.split_indices["val"]
    feature_scale = float(
        np.linalg.norm(
            prepared.features[train_index], axis=-1, keepdims=True
        ).mean()
    )
    if not math.isfinite(feature_scale) or feature_scale <= 0:
        raise RuntimeError(f"Invalid feature normalization scale: {feature_scale}")
    normalized = prepared.features / max(feature_scale, 1e-6)
    tensors = torch.from_numpy(normalized).to(device)
    labels = torch.from_numpy(prepared.labels).to(device)
    train_tensor_index = torch.from_numpy(train_index).to(device)
    val_tensor_index = torch.from_numpy(val_index).to(device)
    train_x = tensors.index_select(0, train_tensor_index)
    train_y = labels.index_select(0, train_tensor_index)
    val_x = tensors.index_select(0, val_tensor_index)
    val_y = prepared.labels[val_index]
    config = experiment_config(args, prepared, device, feature_scale)

    seed_checkpoints: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    for position, seed in enumerate(args.seeds, start=1):
        print(f"Training MIL head seed {seed} ({position}/5)...", flush=True)
        state, best_epoch, best_score, history = train_one_seed(
            train_x,
            train_y,
            val_x,
            val_y,
            seed=seed,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            noise_std=args.noise_std,
            max_epochs=args.max_epochs,
            eval_every=args.eval_every,
            patience=args.patience,
            device=device,
        )
        seed_checkpoints.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "state_dict": state,
            }
        )
        all_history.extend(history)

    checkpoint = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "architecture": "MILHead(512 -> 64 -> 1)",
        "feature_scale": feature_scale,
        "dropout": args.dropout,
        "clips_per_video": CLIPS_PER_VIDEO,
        "frames_per_clip": FRAMES_PER_CLIP,
        "frame_stride": FRAME_STRIDE,
        "top_clips": TOP_CLIPS,
        "threshold": FIXED_THRESHOLD,
        "manifest_sha256": prepared.manifest_sha256,
        "source_audit_sha256": prepared.audit_sha256,
        "seed_checkpoints": seed_checkpoints,
    }
    output_dir = resolve_path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    checkpoint_summary = {
        key: value
        for key, value in checkpoint.items()
        if key != "seed_checkpoints"
    }
    checkpoint_summary["seeds"] = [
        {
            "seed": item["seed"],
            "best_epoch": item["best_epoch"],
            "best_validation_score": item["best_validation_score"],
        }
        for item in seed_checkpoints
    ]
    checkpoint_summary["checkpoint_bytes"] = checkpoint_path.stat().st_size
    write_json(output_dir / "checkpoint.json", checkpoint_summary)
    history_payload = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "selection_metric": config["checkpoint_selection"],
        "selection_split": "val",
        "records": all_history,
    }
    write_json(output_dir / "history.json", history_payload)
    write_csv(
        output_dir / "history.csv",
        all_history,
        (
            "seed",
            "epoch",
            "train_loss",
            "validation_selection_score",
            "validation_accuracy",
            "validation_balanced_accuracy",
            "validation_roc_auc",
            "selected_at_this_epoch",
        ),
    )
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}. Run the run command first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "seed_checkpoints" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    if checkpoint.get("architecture") != "MILHead(512 -> 64 -> 1)":
        raise RuntimeError("Checkpoint architecture does not match this script")
    if len(checkpoint["seed_checkpoints"]) != 5:
        raise RuntimeError("Checkpoint must contain exactly five seed heads")
    if float(checkpoint.get("threshold", -1)) != FIXED_THRESHOLD:
        raise RuntimeError("Checkpoint reporting threshold is not 0.50")
    checkpoint["_device"] = str(device)
    return checkpoint


def predict_ensemble(
    checkpoint: dict[str, Any],
    features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = max(float(checkpoint["feature_scale"]), 1e-6)
    tensor = torch.from_numpy(features.astype(np.float32) / scale).to(device)
    seed_video_probabilities: list[np.ndarray] = []
    seed_clip_logits: list[np.ndarray] = []
    for item in checkpoint["seed_checkpoints"]:
        model = MILHead(dropout=float(checkpoint["dropout"])).to(device)
        model.load_state_dict(item["state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            clip_logits = model.clip_logits(tensor)
            top_count = min(TOP_CLIPS, clip_logits.shape[1])
            video_logits = clip_logits.topk(top_count, dim=1).values.mean(dim=1)
            seed_video_probabilities.append(
                torch.sigmoid(video_logits).detach().cpu().numpy()
            )
            seed_clip_logits.append(clip_logits.detach().cpu().numpy())
    ensemble_probabilities = np.mean(seed_video_probabilities, axis=0)
    ensemble_clip_logits = np.mean(seed_clip_logits, axis=0)
    ensemble_clip_probabilities = 1.0 / (1.0 + np.exp(-ensemble_clip_logits))
    top_indices = np.argsort(ensemble_clip_logits, axis=1)[:, -TOP_CLIPS:][:, ::-1]
    return ensemble_probabilities, ensemble_clip_probabilities, top_indices


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(seed)
    indices_by_class = [
        np.where(labels == label)[0] for label in (0, 1)
    ]
    if any(indices.size == 0 for indices in indices_by_class):
        return {}
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
    )
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(iterations):
        sample = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in indices_by_class
            ]
        )
        sample_metrics = binary_metrics(
            labels[sample], probabilities[sample], FIXED_THRESHOLD
        )
        for name in metric_names:
            value = sample_metrics[name]
            if value is not None:
                values[name].append(float(value))
    return {
        name: (
            [
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            ]
            if samples
            else None
        )
        for name, samples in values.items()
    }


def evaluate_pipeline(
    args: argparse.Namespace,
    prepared: PreparedData,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    output_dir = resolve_path(args.output_dir)
    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint
        else output_dir / "checkpoint.pt"
    )
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get("manifest_sha256") != prepared.manifest_sha256:
        raise RuntimeError("Checkpoint was trained with a different split manifest")

    metrics_by_split: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []
    for split_index, split in enumerate(EVALUATION_SPLITS):
        indices = prepared.split_indices[split]
        probabilities, clip_probabilities, top_indices = predict_ensemble(
            checkpoint, prepared.features[indices], device
        )
        labels = prepared.labels[indices]
        split_metrics = binary_metrics(labels, probabilities, FIXED_THRESHOLD)
        if split in {"test", "ood"}:
            split_metrics["bootstrap_95_ci"] = bootstrap_confidence_intervals(
                labels,
                probabilities,
                iterations=args.bootstrap_iterations,
                seed=20260727 + split_index,
            )
            split_metrics["bootstrap_iterations"] = args.bootstrap_iterations
            split_metrics["bootstrap_method"] = (
                "class-stratified percentile interval"
            )
        metrics_by_split[split] = split_metrics
        for local_index, source_index in enumerate(indices):
            row = prepared.rows[int(source_index)]
            probability = float(probabilities[local_index])
            predicted_label = int(probability >= FIXED_THRESHOLD)
            predictions.append(
                {
                    "relative_path": row.relative_path,
                    "split": split,
                    "actor_group": row.actor_group,
                    "ground_truth": row.label,
                    "ground_truth_name": LABEL_NAMES[row.label],
                    "shoplifting_probability": probability,
                    "predicted_label": predicted_label,
                    "predicted_name": LABEL_NAMES[predicted_label],
                    "correct": predicted_label == row.label,
                    "threshold": FIXED_THRESHOLD,
                    "clip_center_fractions": prepared.fractions.tolist(),
                    "clip_shoplifting_probabilities": clip_probabilities[
                        local_index
                    ].tolist(),
                    "top_2_clip_indices": top_indices[local_index].tolist(),
                }
            )

    metrics_payload = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "reporting_unit": "source_video",
        "fixed_threshold": FIXED_THRESHOLD,
        "video_logit": "mean of top-2 clip logits",
        "ensemble": "mean of five seed video probabilities",
        "model_selection": (
            "validation balanced_accuracy + 0.05 * validation ROC-AUC only"
        ),
        "splits": metrics_by_split,
    }
    write_json(output_dir / "metrics.json", metrics_payload)
    metric_rows: list[dict[str, Any]] = []
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
    )
    for split, metrics in metrics_by_split.items():
        row: dict[str, Any] = {
            "split": split,
            "threshold": FIXED_THRESHOLD,
            "support": metrics["support"],
            "normal_support": metrics["normal_support"],
            "shoplifting_support": metrics["shoplifting_support"],
        }
        for name in metric_names:
            row[name] = metrics[name]
            interval = metrics.get("bootstrap_95_ci", {}).get(name)
            row[f"{name}_ci_low"] = interval[0] if interval else None
            row[f"{name}_ci_high"] = interval[1] if interval else None
        metric_rows.append(row)
    metric_fields = (
        "split",
        "threshold",
        "support",
        "normal_support",
        "shoplifting_support",
        *metric_names,
        *(f"{name}_ci_low" for name in metric_names),
        *(f"{name}_ci_high" for name in metric_names),
    )
    write_csv(output_dir / "metrics.csv", metric_rows, metric_fields)
    write_json(
        output_dir / "predictions.json",
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "fixed_threshold": FIXED_THRESHOLD,
            "predictions": predictions,
        },
    )
    prediction_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        row = prediction.copy()
        row["clip_center_fractions"] = "|".join(
            f"{value:.6f}" for value in prediction["clip_center_fractions"]
        )
        row["clip_shoplifting_probabilities"] = "|".join(
            f"{value:.8f}"
            for value in prediction["clip_shoplifting_probabilities"]
        )
        row["top_2_clip_indices"] = "|".join(
            str(value) for value in prediction["top_2_clip_indices"]
        )
        prediction_rows.append(row)
    write_csv(
        output_dir / "predictions.csv",
        prediction_rows,
        (
            "relative_path",
            "split",
            "actor_group",
            "ground_truth",
            "ground_truth_name",
            "shoplifting_probability",
            "predicted_label",
            "predicted_name",
            "correct",
            "threshold",
            "clip_center_fractions",
            "clip_shoplifting_probabilities",
            "top_2_clip_indices",
        ),
    )
    confusion_rows: list[dict[str, Any]] = []
    for split, metrics in metrics_by_split.items():
        matrix = metrics["confusion_matrix"]["values"]
        for actual_label, actual_name in LABEL_NAMES.items():
            confusion_rows.append(
                {
                    "split": split,
                    "actual_label": actual_name,
                    "predicted_normal": matrix[actual_label][0],
                    "predicted_shoplifting": matrix[actual_label][1],
                }
            )
    write_csv(
        output_dir / "confusion-matrices.csv",
        confusion_rows,
        (
            "split",
            "actual_label",
            "predicted_normal",
            "predicted_shoplifting",
        ),
    )
    render_plots(output_dir, metrics_by_split, predictions, checkpoint)
    return metrics_by_split, predictions, checkpoint


def render_plots(
    output_dir: Path,
    metrics_by_split: dict[str, Any],
    predictions: Sequence[dict[str, Any]],
    checkpoint: dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render metric figures") from exc

    colors = {
        "accuracy": "#2563EB",
        "balanced_accuracy": "#0891B2",
        "f1": "#D97706",
        "roc_auc": "#7C3AED",
        "normal": "#64748B",
        "shoplifting": "#DC2626",
    }
    split_names = list(EVALUATION_SPLITS)
    display_names = ["Train", "Validation", "Test", "OOD", "Qualitative"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    figure.patch.set_facecolor("#F8FAFC")
    figure.suptitle(
        "Shoplifting MIL baseline — fixed 0.50 threshold",
        fontsize=20,
        fontweight="bold",
        color="#0F172A",
    )

    axis = axes[0, 0]
    x = np.arange(len(split_names))
    metric_names = ("accuracy", "balanced_accuracy", "f1", "roc_auc")
    width = 0.19
    for offset, name in enumerate(metric_names):
        values = [
            (
                float(metrics_by_split[split][name])
                if metrics_by_split[split][name] is not None
                else np.nan
            )
            for split in split_names
        ]
        axis.bar(
            x + (offset - 1.5) * width,
            values,
            width,
            label=name.replace("_", " ").title(),
            color=colors[name],
        )
    axis.set_xticks(x, display_names)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title("Video-level metrics")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=2, fontsize=9, frameon=False)

    axis = axes[0, 1]
    normal_support = [
        metrics_by_split[split]["normal_support"] for split in split_names
    ]
    shoplifting_support = [
        metrics_by_split[split]["shoplifting_support"] for split in split_names
    ]
    axis.bar(x, normal_support, color=colors["normal"], label="Normal")
    axis.bar(
        x,
        shoplifting_support,
        bottom=normal_support,
        color=colors["shoplifting"],
        label="Shoplifting",
    )
    axis.set_xticks(x, display_names)
    axis.set_ylabel("Source videos")
    axis.set_title("Evaluated support")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)

    axis = axes[1, 0]
    rng = np.random.default_rng(20260727)
    for split_index, split in enumerate(("test", "ood")):
        split_predictions = [
            row for row in predictions if row["split"] == split
        ]
        for label, marker, color in (
            (0, "o", colors["normal"]),
            (1, "^", colors["shoplifting"]),
        ):
            values = [
                float(row["shoplifting_probability"])
                for row in split_predictions
                if row["ground_truth"] == label
            ]
            jitter = rng.normal(0, 0.035, len(values))
            axis.scatter(
                np.full(len(values), split_index) + jitter,
                values,
                s=58,
                alpha=0.82,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                label=(
                    LABEL_NAMES[label]
                    if split_index == 0
                    else None
                ),
            )
    axis.axhline(
        FIXED_THRESHOLD,
        color="#111827",
        linewidth=1.4,
        linestyle="--",
        label="Threshold 0.50",
    )
    axis.set_xticks((0, 1), ("Test", "OOD"))
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Shoplifting probability")
    axis.set_title("Held-out probability distribution")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3, fontsize=9)

    axis = axes[1, 1]
    axis.axis("off")
    seed_lines = "\n".join(
        f"  Seed {item['seed']}: epoch {item['best_epoch']}, "
        f"val score {item['best_validation_score']:.4f}"
        for item in checkpoint["seed_checkpoints"]
    )
    test_metrics = metrics_by_split["test"]
    ood_metrics = metrics_by_split["ood"]
    summary = (
        "Method\n"
        "  Frozen R3D-18 Kinetics embeddings\n"
        "  7 clips/video · 16 frames · stride 4\n"
        "  MLP 512→64→1 · mean top-2 clip logits\n"
        "  Five-seed probability ensemble\n"
        "  Model selection: validation metrics only\n\n"
        "Held-out results\n"
        f"  Test accuracy: {test_metrics['accuracy']:.3f}\n"
        f"  Test balanced accuracy: {test_metrics['balanced_accuracy']:.3f}\n"
        f"  OOD accuracy: {ood_metrics['accuracy']:.3f}\n"
        f"  OOD balanced accuracy: {ood_metrics['balanced_accuracy']:.3f}\n\n"
        "Selected checkpoints\n"
        f"{seed_lines}"
    )
    axis.text(
        0.02,
        0.98,
        summary,
        va="top",
        ha="left",
        fontsize=10.5,
        linespacing=1.35,
        family="monospace",
        color="#0F172A",
        bbox={
            "boxstyle": "round,pad=0.9",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CBD5E1",
        },
    )
    figure.savefig(
        output_dir / "metrics-dashboard.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(split_names), figsize=(18, 4.3), constrained_layout=True
    )
    figure.patch.set_facecolor("#F8FAFC")
    figure.suptitle(
        "Confusion matrices — counts at threshold 0.50",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
    )
    for axis, split, display_name in zip(axes, split_names, display_names):
        matrix = np.asarray(
            metrics_by_split[split]["confusion_matrix"]["values"], dtype=np.int64
        )
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        for row_index in range(2):
            for column_index in range(2):
                value = int(matrix[row_index, column_index])
                axis.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color=(
                        "white"
                        if value > max(1, int(matrix.max())) * 0.55
                        else "#0F172A"
                    ),
                )
        axis.set_title(display_name, fontweight="bold")
        axis.set_xticks((0, 1), ("Normal", "Shoplifting"), rotation=25)
        axis.set_yticks((0, 1), ("Normal", "Shoplifting"))
        axis.set_xlabel("Predicted")
        if axis is axes[0]:
            axis.set_ylabel("Ground truth")
    figure.savefig(
        output_dir / "confusion-matrices.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def demo_window_starts(
    frame_count: int,
    fps: float,
    stride_seconds: float,
    max_windows: int,
) -> list[int]:
    span = (FRAMES_PER_CLIP - 1) * FRAME_STRIDE
    max_start = max(0, frame_count - 1 - span)
    step = max(1, int(round(fps * stride_seconds)))
    starts = list(range(0, max_start + 1, step))
    if not starts or starts[-1] != max_start:
        starts.append(max_start)
    if len(starts) > max_windows:
        starts = sorted(
            {
                int(round(value))
                for value in np.linspace(0, max_start, max_windows)
            }
        )
    return starts


def decode_clip_at_start(
    capture: cv2.VideoCapture,
    start: int,
    frame_count: int,
    preprocess: nn.Module,
) -> torch.Tensor:
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[np.ndarray] = []
    target_offsets = {
        index * FRAME_STRIDE for index in range(FRAMES_PER_CLIP)
    }
    final_offset = (FRAMES_PER_CLIP - 1) * FRAME_STRIDE
    last_frame: np.ndarray | None = None
    for offset in range(final_offset + 1):
        ok, frame = capture.read()
        if not ok or frame is None:
            if last_frame is None:
                raise RuntimeError(f"Could not decode demo clip at frame {start}")
            frame = last_frame
        last_frame = frame
        if offset in target_offsets:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    while len(frames) < FRAMES_PER_CLIP:
        if last_frame is None:
            raise RuntimeError("Demo clip has no decodable frames")
        frames.append(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
    tensor = (
        torch.from_numpy(np.stack(frames))
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return preprocess(tensor)


def extract_demo_window_embeddings(
    video_path: Path,
    starts: Sequence[int],
    device: torch.device,
    weights_file: Path | None,
    batch_size: int,
) -> np.ndarray:
    model, preprocess = load_r3d_backbone(device, weights_file)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open demo source: {video_path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    features: list[np.ndarray] = []
    pending: list[torch.Tensor] = []

    def flush() -> None:
        if not pending:
            return
        batch = torch.stack(pending).to(device, non_blocking=device.type == "cuda")
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            output = model(batch).float().cpu().numpy()
        features.extend(output)
        pending.clear()

    try:
        for start in starts:
            pending.append(
                decode_clip_at_start(capture, int(start), frame_count, preprocess)
            )
            if len(pending) >= batch_size:
                flush()
        flush()
    finally:
        capture.release()
    return np.asarray(features, dtype=np.float32)


def rolling_window_probabilities(
    checkpoint: dict[str, Any],
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    scale = max(float(checkpoint["feature_scale"]), 1e-6)
    tensor = torch.from_numpy(features / scale).to(device)
    seed_probabilities: list[np.ndarray] = []
    for item in checkpoint["seed_checkpoints"]:
        model = MILHead(dropout=float(checkpoint["dropout"])).to(device)
        model.load_state_dict(item["state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            logits = model.clip_logits(tensor[:, None, :]).squeeze(1)
            seed_probabilities.append(
                torch.sigmoid(logits).detach().cpu().numpy()
            )
    return np.mean(seed_probabilities, axis=0)


def fit_output_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    if max_width <= 0 or width <= max_width:
        output_width, output_height = width, height
    else:
        scale = max_width / width
        output_width = max_width
        output_height = int(round(height * scale))
    output_width = max(2, output_width - output_width % 2)
    output_height = max(2, output_height - output_height % 2)
    return output_width, output_height


def track_color(track_id: int) -> tuple[int, int, int]:
    palette = (
        (214, 203, 0),
        (192, 229, 45),
        (255, 181, 62),
        (214, 133, 40),
        (246, 207, 95),
    )
    return palette[track_id % len(palette)]


def draw_tracking_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.42, min(0.55, frame.shape[1] / 1450))
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = origin
    x = max(0, min(x, frame.shape[1] - text_width - 8))
    top = max(0, y - text_height - baseline - 7)
    cv2.rectangle(
        frame,
        (x, top),
        (x + text_width + 8, min(frame.shape[0] - 1, y + 2)),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 4, max(text_height + 1, y - baseline - 2)),
        font,
        scale,
        (12, 18, 24),
        1,
        cv2.LINE_AA,
    )


def draw_yolo_person_tracks(
    frame: np.ndarray,
    result: Any,
    track_history: dict[int, list[tuple[int, int]]],
) -> int:
    """Draw person perception only; the MIL head supplies behavior probability."""

    boxes = result.boxes
    if boxes is None or not len(boxes):
        return 0
    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    track_ids = (
        boxes.id.detach().cpu().numpy().astype(int)
        if boxes.id is not None
        else np.arange(len(coordinates), dtype=int)
    )
    for box, confidence, track_id in zip(coordinates, confidences, track_ids):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        x1 = int(np.clip(x1, 0, frame.shape[1] - 1))
        x2 = int(np.clip(x2, 0, frame.shape[1] - 1))
        y1 = int(np.clip(y1, 0, frame.shape[0] - 1))
        y2 = int(np.clip(y2, 0, frame.shape[0] - 1))
        color = track_color(int(track_id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        draw_tracking_label(
            frame,
            f"YOLO PERSON #{int(track_id):02d}  {float(confidence):.2f}",
            (x1, max(25, y1)),
            color,
        )
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        history = track_history[int(track_id)]
        history.append(center)
        del history[:-24]
        if len(history) > 1:
            cv2.polylines(
                frame,
                [np.asarray(history, dtype=np.int32).reshape((-1, 1, 2))],
                False,
                color,
                2,
                cv2.LINE_AA,
            )
    return len(coordinates)


def draw_demo_overlay(
    frame: np.ndarray,
    rolling_probability: float,
    video_probability: float,
    relative_path: str,
    test_metrics: dict[str, Any],
    frame_number: int,
    frame_count: int,
    perception_enabled: bool,
    person_count: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    panel_height = min(height - 56, 184)
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (10, 18, 30), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.46, min(0.72, width / 1050))
    color = (244, 247, 250)
    cv2.putText(
        frame,
        "EDCOSYS | HELD-OUT SHOPLIFTING DEMO",
        (16, 25),
        font,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"GT: Shoplifting | Test split | Source: {relative_path}",
        (16, 51),
        font,
        max(0.42, scale * 0.82),
        (210, 221, 232),
        1,
        cv2.LINE_AA,
    )
    predicted_name = LABEL_NAMES[int(video_probability >= FIXED_THRESHOLD)]
    perception_text = (
        f"YOLO26s + ByteTrack: perception ({person_count} people) | "
        "R3D18-MIL: shoplifting probability"
        if perception_enabled
        else "R3D18-MIL: shoplifting probability | YOLO overlay: model absent"
    )
    cv2.putText(
        frame,
        perception_text,
        (16, 77),
        font,
        max(0.42, scale * 0.78),
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            f"MIL rolling p: {rolling_probability:.3f} | Video p: "
            f"{video_probability:.3f} | Threshold: {FIXED_THRESHOLD:.2f} | "
            f"Prediction: {predicted_name}"
        ),
        (16, 103),
        font,
        max(0.42, scale * 0.78),
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            f"Internal test: accuracy {test_metrics['accuracy'] * 100:.1f}% | "
            f"balanced accuracy {test_metrics['balanced_accuracy'] * 100:.1f}% | "
            f"n={test_metrics['support']}"
        ),
        (16, 129),
        font,
        max(0.42, scale * 0.78),
        (190, 215, 225),
        1,
        cv2.LINE_AA,
    )
    bar_left, bar_right = 16, max(26, width - 16)
    bar_top, bar_bottom = 145, min(panel_height - 10, 171)
    cv2.rectangle(
        frame, (bar_left, bar_top), (bar_right, bar_bottom), (67, 78, 91), -1
    )
    fill = int(
        round((bar_right - bar_left) * np.clip(rolling_probability, 0.0, 1.0))
    )
    bar_color = (
        (52, 73, 230)
        if rolling_probability >= FIXED_THRESHOLD
        else (87, 176, 86)
    )
    cv2.rectangle(
        frame, (bar_left, bar_top), (bar_left + fill, bar_bottom), bar_color, -1
    )
    threshold_x = int(
        round(bar_left + (bar_right - bar_left) * FIXED_THRESHOLD)
    )
    cv2.line(
        frame,
        (threshold_x, bar_top - 3),
        (threshold_x, bar_bottom + 3),
        (255, 255, 255),
        2,
    )
    footer_overlay = frame.copy()
    cv2.rectangle(
        footer_overlay,
        (0, max(0, height - 50)),
        (width, height),
        (10, 18, 30),
        -1,
    )
    cv2.addWeighted(footer_overlay, 0.84, frame, 0.16, 0, frame)
    cv2.putText(
        frame,
        "MNNIT Allahabad Shoplifting Dataset | CC BY 4.0",
        (12, height - 29),
        font,
        max(0.38, scale * 0.68),
        (220, 229, 238),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "DOI 10.17632/r3yjf35hzr.1 | actor-disjoint research evaluation",
        (12, height - 10),
        font,
        max(0.36, scale * 0.64),
        (190, 205, 219),
        1,
        cv2.LINE_AA,
    )
    progress = safe_divide(frame_number + 1, frame_count)
    cv2.putText(
        frame,
        f"{progress * 100:5.1f}%",
        (max(8, width - 74), height - 10),
        font,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    return frame


def render_demo(
    args: argparse.Namespace,
    prepared: PreparedData,
    device: torch.device,
    metrics_by_split: dict[str, Any],
    predictions: Sequence[dict[str, Any]],
    checkpoint: dict[str, Any],
) -> Path:
    data_root = resolve_path(args.data_root)
    if args.video and str(args.video).casefold() != "auto":
        requested = str(args.video).replace("\\", "/")
        requested_path = Path(requested)
        if requested_path.is_absolute():
            relative_path = requested_path.resolve().relative_to(data_root).as_posix()
        else:
            relative_path = PurePosixPath(requested).as_posix()
        candidates = [
            row
            for row in prepared.rows
            if row.relative_path.casefold() == relative_path.casefold()
        ]
        if not candidates:
            raise RuntimeError(f"Demo video is absent from the manifest: {relative_path}")
        record = candidates[0]
    else:
        record = sorted(
            [
                row
                for row in prepared.rows
                if row.split == "test" and row.label == 1
            ],
            key=lambda row: natural_video_key(row.relative_path),
        )[0]
    if record.split != "test" or record.label != 1:
        raise RuntimeError(
            "The demo source must be a held-out test Shoplifting video"
        )
    official_prediction = next(
        (
            row
            for row in predictions
            if row["relative_path"].casefold() == record.relative_path.casefold()
            and row["split"] == "test"
        ),
        None,
    )
    if official_prediction is None:
        raise RuntimeError("Selected demo source has no test prediction")
    video_probability = float(official_prediction["shoplifting_probability"])
    video_path = data_root / Path(record.relative_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open demo source: {video_path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError("Demo source has invalid FPS or frame count")
    starts = demo_window_starts(
        frame_count,
        fps,
        args.demo_stride_seconds,
        args.demo_max_windows,
    )
    print(
        f"Extracting {len(starts)} rolling windows from {record.relative_path}...",
        flush=True,
    )
    window_features = extract_demo_window_embeddings(
        video_path,
        starts,
        device,
        resolve_path(args.weights_file) if args.weights_file else None,
        args.embedding_batch_size,
    )
    probabilities = rolling_window_probabilities(
        checkpoint, window_features, device
    )
    span = (FRAMES_PER_CLIP - 1) * FRAME_STRIDE + 1
    centers = np.asarray(
        [start + (span - 1) / 2 for start in starts], dtype=np.float64
    )

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is required for H.264 demo encoding"
        ) from exc
    destination = resolve_path(args.demo_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.mp4")
    output_width, output_height = fit_output_size(width, height, args.demo_max_width)
    yolo_model_path = resolve_path(args.yolo_model)
    yolo_model: Any | None = None
    if yolo_model_path.is_file() and not args.disable_yolo_overlay:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                f"{yolo_model_path} exists, but ultralytics is unavailable"
            ) from exc
        yolo_model = YOLO(str(yolo_model_path))
        print(
            f"Enabling YOLO26s + ByteTrack person overlay: {yolo_model_path}",
            flush=True,
        )
    elif not args.disable_yolo_overlay:
        print(
            f"YOLO overlay skipped; model is absent: {yolo_model_path}",
            flush=True,
        )
    track_history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    yolo_device: str | int = "cpu"
    if device.type == "cuda":
        yolo_device = device.index if device.index is not None else 0
    writer = imageio_ffmpeg.write_frames(
        str(temporary),
        (output_width, output_height),
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        fps=fps,
        quality=None,
        codec="libx264",
        macro_block_size=2,
        ffmpeg_log_level="warning",
        output_params=[
            "-crf",
            "20",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
        ],
    )
    writer.send(None)
    capture = cv2.VideoCapture(str(video_path))
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            rolling_probability = float(
                np.interp(
                    written,
                    centers,
                    probabilities,
                    left=probabilities[0],
                    right=probabilities[-1],
                )
            )
            if frame.shape[1] != output_width or frame.shape[0] != output_height:
                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
            person_count = 0
            if yolo_model is not None:
                result = yolo_model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=args.yolo_conf,
                    imgsz=args.yolo_imgsz,
                    device=yolo_device,
                    verbose=False,
                )[0]
                person_count = draw_yolo_person_tracks(
                    frame, result, track_history
                )
            frame = draw_demo_overlay(
                frame,
                rolling_probability,
                video_probability,
                record.relative_path,
                metrics_by_split["test"],
                written,
                frame_count,
                yolo_model is not None,
                person_count,
            )
            writer.send(
                np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).tobytes()
            )
            written += 1
    finally:
        capture.release()
        writer.close()
    if written == 0 or not temporary.is_file():
        raise RuntimeError("Demo encoder produced no frames")
    temporary.replace(destination)

    output_dir = resolve_path(args.output_dir)
    window_rows = [
        {
            "window_index": index,
            "start_frame": int(start),
            "center_frame": float(center),
            "center_seconds": float(center / fps),
            "rolling_shoplifting_probability": float(probability),
        }
        for index, (start, center, probability) in enumerate(
            zip(starts, centers, probabilities)
        )
    ]
    write_csv(
        output_dir / "demo-windows.csv",
        window_rows,
        (
            "window_index",
            "start_frame",
            "center_frame",
            "center_seconds",
            "rolling_shoplifting_probability",
        ),
    )
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "selection_policy": (
            "First numeric held-out test Shoplifting source when --video=auto; "
            "selection is independent of model confidence and correctness"
        ),
        "source_relative_path": record.relative_path,
        "split": record.split,
        "ground_truth": LABEL_NAMES[record.label],
        "video_shoplifting_probability": video_probability,
        "video_prediction": LABEL_NAMES[
            int(video_probability >= FIXED_THRESHOLD)
        ],
        "correct": int(video_probability >= FIXED_THRESHOLD) == record.label,
        "fixed_threshold": FIXED_THRESHOLD,
        "test_accuracy": metrics_by_split["test"]["accuracy"],
        "test_balanced_accuracy": metrics_by_split["test"]["balanced_accuracy"],
        "test_support": metrics_by_split["test"]["support"],
        "rolling_windows": len(starts),
        "frames_per_window": FRAMES_PER_CLIP,
        "frame_stride": FRAME_STRIDE,
        "output": str(destination),
        "output_codec": "H.264 / libx264",
        "output_pixel_format": "yuv420p",
        "perception_overlay": {
            "enabled": yolo_model is not None,
            "model": str(yolo_model_path) if yolo_model is not None else None,
            "task": "person detection only",
            "tracker": "ByteTrack" if yolo_model is not None else None,
            "confidence_threshold": args.yolo_conf,
            "image_size": args.yolo_imgsz,
        },
        "behavior_probability_model": "R3D-18 embeddings + five-seed MIL heads",
        "dataset_attribution": {
            "name": "MNNIT Allahabad Shoplifting Dataset",
            "license": "CC BY 4.0",
            "doi": "10.17632/r3yjf35hzr.1",
        },
        "output_frames": written,
        "output_fps": fps,
        "output_width": output_width,
        "output_height": output_height,
    }
    write_json(output_dir / "demo-summary.json", summary)
    print(f"Wrote H.264 demo: {destination}", flush=True)
    return destination


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--cache-metadata", type=Path, default=DEFAULT_CACHE_METADATA
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to OUTPUT_DIR/checkpoint.pt.",
    )
    parser.add_argument(
        "--weights-file",
        type=Path,
        default=None,
        help="Optional local official TorchVision R3D-18 state dict.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seeds", type=int, nargs=5, default=list(DEFAULT_SEEDS)
    )
    parser.add_argument("--max-epochs", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.002)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Replace the embedding cache after the source audit.",
    )


def add_demo_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--video",
        default="auto",
        help="Held-out test Shoplifting relative path; auto selects the first numeric ID.",
    )
    parser.add_argument(
        "--demo-output", type=Path, default=DEFAULT_DEMO_OUTPUT
    )
    parser.add_argument("--demo-stride-seconds", type=float, default=0.5)
    parser.add_argument("--demo-max-windows", type=int, default=240)
    parser.add_argument("--demo-max-width", type=int, default=960)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument(
        "--disable-yolo-overlay",
        action="store_true",
        help="Render the MIL demo without optional YOLO26s + ByteTrack boxes.",
    )


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "embedding_batch_size",
        "bootstrap_iterations",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "max_epochs") and args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    if hasattr(args, "eval_every") and args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if hasattr(args, "patience") and args.patience <= 0:
        raise ValueError("--patience must be positive")
    if hasattr(args, "demo_stride_seconds") and args.demo_stride_seconds <= 0:
        raise ValueError("--demo-stride-seconds must be positive")
    if hasattr(args, "demo_max_windows") and args.demo_max_windows <= 0:
        raise ValueError("--demo-max-windows must be positive")
    if hasattr(args, "yolo_conf") and not 0 < args.yolo_conf <= 1:
        raise ValueError("--yolo-conf must be in (0, 1]")
    if hasattr(args, "yolo_imgsz") and args.yolo_imgsz <= 0:
        raise ValueError("--yolo-imgsz must be positive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train, evaluate, and demonstrate the five-seed weakly supervised "
            "R3D-18 MIL shoplifting baseline."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Audit, extract/reuse embeddings, train, evaluate, and render demo."
    )
    add_shared_arguments(run_parser)
    add_training_arguments(run_parser)
    add_demo_arguments(run_parser)
    run_parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Finish after metrics and figures.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Recompute all fixed-threshold metrics and figures."
    )
    add_shared_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Replace the embedding cache before evaluation.",
    )

    demo_parser = subparsers.add_parser(
        "demo", help="Evaluate and render an annotated held-out test MP4."
    )
    add_shared_arguments(demo_parser)
    add_demo_arguments(demo_parser)
    demo_parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Replace the embedding cache before rendering.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)
    prepared = prepare_data(args, device)
    if args.command == "run":
        train_pipeline(args, prepared, device)
        metrics, predictions, checkpoint = evaluate_pipeline(
            args, prepared, device
        )
        if not args.skip_demo:
            render_demo(
                args,
                prepared,
                device,
                metrics,
                predictions,
                checkpoint,
            )
    elif args.command == "evaluate":
        evaluate_pipeline(args, prepared, device)
    elif args.command == "demo":
        metrics, predictions, checkpoint = evaluate_pipeline(
            args, prepared, device
        )
        render_demo(
            args,
            prepared,
            device,
            metrics,
            predictions,
            checkpoint,
        )
    else:  # argparse enforces this; retain a defensive failure for API use
        parser.error(f"Unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
