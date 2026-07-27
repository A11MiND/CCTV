"""Frozen MViT-V2-S dense-clip MIL experiment for shoplifting classification.

This experiment deliberately keeps the internal test split sealed by default.
``run`` extracts or reuses embeddings for train/validation videos only, trains
five seeds for each requested MIL pooling method, selects checkpoints and the
pooling method using validation metrics only, and writes:

    output/shoplifting-mvit-experiment/summary.json
    output/shoplifting-mvit-experiment/checkpoint.pt
    output/shoplifting-mvit-experiment/predictions.json

Only ``evaluate --include-test`` may extract, score, or emit predictions for the
reviewed test split.  OOD and qualitative splits are outside this experiment.

The frozen feature extractor is TorchVision MViT-V2-S with Kinetics-400
weights.  Every source video is represented by 15 or 21 edge-inclusive dense
clips.  Each clip contains 16 full frames sampled with temporal stride four.
Embeddings are cached below ``data/cache`` and aligned by manifest path.

Examples, from the repository root:

    .venv-yolo\\Scripts\\python.exe scripts\\shoplifting_mvit_experiment.py run

    .venv-yolo\\Scripts\\python.exe scripts\\shoplifting_mvit_experiment.py run \
        --clips-per-video 21

    .venv-yolo\\Scripts\\python.exe scripts\\shoplifting_mvit_experiment.py \
        evaluate --include-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

import shoplifting_mil_baseline as baseline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "shoplifting-video-dataset"
DEFAULT_MANIFEST = (
    ROOT / "docs" / "results" / "shoplifting" / "actor-clothing-disjoint-split.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "output" / "shoplifting-mvit-experiment"
DEFAULT_SEEDS = (11, 23, 37, 51, 79)
POOLING_CHOICES = ("top1", "top2", "logsumexp", "attention")
FRAMES_PER_CLIP = 16
FRAME_STRIDE = 4
EMBEDDING_DIMENSION = 768
HIDDEN_DIMENSION = 128
FIXED_THRESHOLD = 0.50
SCRIPT_VERSION = "1.0.0"

if (ROOT / "data" / "torch-cache").is_dir():
    os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "torch-cache"))


class MViTMILHead(nn.Module):
    """Clip classifier with selectable multiple-instance aggregation."""

    def __init__(
        self,
        pooling: str,
        dropout: float = 0.30,
        logsumexp_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if pooling not in POOLING_CHOICES:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        if logsumexp_temperature <= 0:
            raise ValueError("logsumexp_temperature must be positive")
        self.pooling = pooling
        self.logsumexp_temperature = float(logsumexp_temperature)
        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING_DIMENSION, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIMENSION, 1),
        )
        self.attention = (
            nn.Sequential(
                nn.Linear(EMBEDDING_DIMENSION, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            if pooling == "attention"
            else None
        )

    def clip_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings).squeeze(-1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.clip_logits(embeddings)
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
        weights = torch.softmax(self.attention(embeddings).squeeze(-1), dim=1)
        return (weights * logits).sum(dim=1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def default_cache_path(clips_per_video: int) -> Path:
    return (
        ROOT
        / "data"
        / "cache"
        / f"mvit-v2-s-dense-{clips_per_video}-embeddings.npz"
    )


def strict_json_write(path: Path, payload: Any) -> None:
    baseline.write_json(path, payload)


def edge_inclusive_starts(frame_count: int, clips_per_video: int) -> np.ndarray:
    """Return exactly N starts, including both valid temporal edges."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    span = (FRAMES_PER_CLIP - 1) * FRAME_STRIDE
    max_start = max(0, frame_count - 1 - span)
    starts = np.rint(
        np.linspace(0, max_start, clips_per_video, dtype=np.float64)
    ).astype(np.int64)
    starts[0] = 0
    starts[-1] = max_start
    if np.any(np.diff(starts) < 0):
        raise RuntimeError("Dense clip starts are not monotonic")
    return starts


def iter_preprocessed_clips(
    path: Path,
    starts: np.ndarray,
    preprocess: nn.Module,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Decode a video once and yield completed clips in temporal order."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Invalid frame count for {path}")

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
    final_target = max(targets)
    try:
        for frame_index in range(final_target + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Could not decode frame {frame_index} from {path}"
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
                buffer[slot] = rgb
                touched.add(clip_index)
            for clip_index in sorted(touched):
                buffer = buffers[clip_index]
                if buffer is None or any(item is None for item in buffer):
                    continue
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
        raise RuntimeError(f"Incomplete clips in {path}: {missing}")


def load_mvit_backbone(
    device: torch.device,
    weights_file: Path | None,
) -> tuple[nn.Module, nn.Module]:
    weights = MViT_V2_S_Weights.KINETICS400_V1
    if weights_file is None:
        try:
            model = mvit_v2_s(weights=weights, progress=True)
        except Exception as exc:
            raise RuntimeError(
                "Could not load cached TorchVision MViT-V2-S Kinetics-400 "
                "weights. Populate data/torch-cache or pass --weights-file."
            ) from exc
    else:
        model = mvit_v2_s(weights=None)
        payload = torch.load(weights_file, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unsupported MViT state dict: {weights_file}")
        state = {
            str(key).removeprefix("module."): value
            for key, value in payload.items()
            if isinstance(value, torch.Tensor)
        }
        model.load_state_dict(state, strict=True)
    model.head = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval().to(device)
    return model, weights.transforms()


def load_cache(
    cache_path: Path,
    clips_per_video: int,
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
            "frames_per_clip",
            "frame_stride",
        }
        missing = required - set(payload.files)
        if missing:
            raise RuntimeError(f"MViT cache is missing arrays: {sorted(missing)}")
        features = payload["features"].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
        paths = payload["paths"].astype(str)
        splits = payload["splits"].astype(str)
        start_frames = payload["start_frames"].astype(np.int64)
        cached_frames = int(np.asarray(payload["frames_per_clip"]).item())
        cached_stride = int(np.asarray(payload["frame_stride"]).item())
    if features.ndim != 3 or features.shape[1:] != (
        clips_per_video,
        EMBEDDING_DIMENSION,
    ):
        raise RuntimeError(
            f"MViT cache has incompatible feature shape: {features.shape}"
        )
    expected_rows = features.shape[0]
    if (
        labels.shape != (expected_rows,)
        or paths.shape != (expected_rows,)
        or splits.shape != (expected_rows,)
        or start_frames.shape != (expected_rows, clips_per_video)
    ):
        raise RuntimeError("MViT cache arrays have inconsistent row counts")
    if cached_frames != FRAMES_PER_CLIP or cached_stride != FRAME_STRIDE:
        raise RuntimeError("MViT cache temporal configuration is incompatible")
    if not np.isfinite(features).all():
        raise RuntimeError("MViT cache contains non-finite embeddings")
    records: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(paths):
        key = str(path).replace("\\", "/").casefold()
        if key in records:
            raise RuntimeError(f"Duplicate MViT cache path: {path}")
        records[key] = {
            "relative_path": str(path).replace("\\", "/"),
            "label": int(labels[index]),
            "split": str(splits[index]),
            "features": features[index],
            "start_frames": start_frames[index],
        }
    return records


def write_cache(
    cache_path: Path,
    records: dict[str, dict[str, Any]],
    clips_per_video: int,
    manifest_sha256: str,
    extraction_seconds: float,
) -> None:
    ordered = sorted(
        records.values(),
        key=lambda item: baseline.natural_video_key(item["relative_path"]),
    )
    features = np.stack([item["features"] for item in ordered]).astype(np.float32)
    labels = np.asarray([item["label"] for item in ordered], dtype=np.int64)
    paths = np.asarray([item["relative_path"] for item in ordered])
    splits = np.asarray([item["split"] for item in ordered])
    starts = np.stack([item["start_frames"] for item in ordered]).astype(np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        features=features,
        labels=labels,
        paths=paths,
        splits=splits,
        start_frames=starts,
        frames_per_clip=np.asarray(FRAMES_PER_CLIP, dtype=np.int64),
        frame_stride=np.asarray(FRAME_STRIDE, dtype=np.int64),
    )
    temporary.replace(cache_path)
    strict_json_write(
        cache_path.with_suffix(".json"),
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": utc_now(),
            "feature_extractor": "TorchVision MViT-V2-S Kinetics-400",
            "embedding_dimension": EMBEDDING_DIMENSION,
            "videos": len(ordered),
            "splits": dict(
                sorted(
                    {
                        split: sum(item["split"] == split for item in ordered)
                        for split in {item["split"] for item in ordered}
                    }.items()
                )
            ),
            "clips_per_video": clips_per_video,
            "edge_inclusive": True,
            "frames_per_clip": FRAMES_PER_CLIP,
            "frame_stride": FRAME_STRIDE,
            "manifest_sha256": manifest_sha256,
            "last_extraction_seconds": extraction_seconds,
        },
    )


def extract_missing_embeddings(
    missing_rows: Sequence[baseline.ManifestRow],
    data_root: Path,
    clips_per_video: int,
    device: torch.device,
    weights_file: Path | None,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    model, preprocess = load_mvit_backbone(device, weights_file)
    results: dict[str, dict[str, Any]] = {}
    pending: list[torch.Tensor] = []
    pending_locations: list[tuple[str, int]] = []
    output_features: dict[str, np.ndarray] = {
        row.relative_path: np.zeros(
            (clips_per_video, EMBEDDING_DIMENSION), dtype=np.float32
        )
        for row in missing_rows
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
            embeddings = model(batch).float().cpu().numpy()
        if embeddings.shape[1:] != (EMBEDDING_DIMENSION,):
            raise RuntimeError(
                f"MViT produced unexpected embedding shape: {embeddings.shape}"
            )
        for embedding, (relative_path, clip_index) in zip(
            embeddings, pending_locations
        ):
            output_features[relative_path][clip_index] = embedding
        pending.clear()
        pending_locations.clear()

    for video_index, row in enumerate(missing_rows):
        video_path = data_root / Path(row.relative_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
        starts = edge_inclusive_starts(frame_count, clips_per_video)
        for clip_index, tensor in iter_preprocessed_clips(
            video_path, starts, preprocess
        ):
            pending.append(tensor)
            pending_locations.append((row.relative_path, clip_index))
            if len(pending) >= batch_size:
                flush()
        flush()
        features = output_features[row.relative_path]
        if not np.isfinite(features).all() or np.any(
            np.linalg.norm(features, axis=-1) <= 0
        ):
            raise RuntimeError(f"Invalid MViT embeddings for {row.relative_path}")
        results[row.relative_path.casefold()] = {
            "relative_path": row.relative_path,
            "label": row.label,
            "split": row.split,
            "features": features,
            "start_frames": starts,
        }
        print(
            f"MViT embeddings {video_index + 1}/{len(missing_rows)}: "
            f"{row.relative_path}",
            flush=True,
        )
    flush()
    return results


def ensure_embeddings(
    args: argparse.Namespace,
    rows: Sequence[baseline.ManifestRow],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, bool]:
    clips_per_video = args.clips_per_video
    cache_path = (
        resolve_path(args.cache)
        if args.cache is not None
        else default_cache_path(clips_per_video)
    )
    data_root = resolve_path(args.data_root)
    manifest_path = resolve_path(args.split_manifest)
    records = {} if args.force_extract else load_cache(
        cache_path, clips_per_video
    )
    manifest_by_key = {row.relative_path.casefold(): row for row in rows}
    for key, record in records.items():
        manifest_row = manifest_by_key.get(key)
        if manifest_row is None:
            # A cache may contain the explicitly opened test split from a prior
            # evaluate --include-test run. Default run/evaluate commands ignore
            # those rows and never emit their predictions.
            continue
        if (
            record["label"] != manifest_row.label
            or record["split"] != manifest_row.split
        ):
            raise RuntimeError(
                f"Cached label/split mismatch for {record['relative_path']}"
            )
    missing_rows = [
        row for row in rows if row.relative_path.casefold() not in records
    ]
    cache_reused = bool(records) and not missing_rows
    if missing_rows:
        started = time.perf_counter()
        extracted = extract_missing_embeddings(
            missing_rows=missing_rows,
            data_root=data_root,
            clips_per_video=clips_per_video,
            device=device,
            weights_file=(
                resolve_path(args.weights_file) if args.weights_file else None
            ),
            batch_size=args.embedding_batch_size,
        )
        records.update(extracted)
        write_cache(
            cache_path,
            records,
            clips_per_video,
            baseline.sha256_file(manifest_path),
            time.perf_counter() - started,
        )
    aligned = [records[row.relative_path.casefold()] for row in rows]
    features = np.stack([item["features"] for item in aligned]).astype(np.float32)
    labels = np.asarray([item["label"] for item in aligned], dtype=np.int64)
    starts = np.stack([item["start_frames"] for item in aligned]).astype(np.int64)
    return features, labels, starts, cache_path, cache_reused


def validation_score(metrics: dict[str, Any]) -> float:
    auc = metrics["roc_auc"]
    if auc is None:
        return -math.inf
    return float(metrics["balanced_accuracy"]) + 0.05 * float(auc)


def train_seed(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    *,
    pooling: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, float]:
    baseline.seed_everything(seed)
    model = MViTMILHead(
        pooling=pooling,
        dropout=args.dropout,
        logsumexp_temperature=args.logsumexp_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    negative_count = int((train_y == 0).sum().item())
    positive_count = int((train_y == 1).sum().item())
    if not negative_count or not positive_count:
        raise RuntimeError("Training split requires both classes")
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
        noisy = train_x + torch.randn_like(train_x) * args.noise_std
        loss = loss_function(model(noisy), train_y.float())
        loss.backward()
        optimizer.step()
        if epoch % args.eval_every:
            continue
        model.eval()
        with torch.inference_mode():
            val_probabilities = torch.sigmoid(model(val_x)).cpu().numpy()
        metrics = baseline.binary_metrics(
            val_y, val_probabilities, FIXED_THRESHOLD
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
            f"No valid validation checkpoint for pooling={pooling}, seed={seed}"
        )
    return best_state, best_epoch, best_score


def ensemble_probabilities(
    states: Sequence[dict[str, Any]],
    pooling: str,
    features: np.ndarray,
    feature_scale: float,
    args_or_checkpoint: Any,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(
        features.astype(np.float32) / max(feature_scale, 1e-6)
    ).to(device)
    video_probabilities: list[np.ndarray] = []
    clip_probabilities: list[np.ndarray] = []
    dropout = float(
        args_or_checkpoint.dropout
        if hasattr(args_or_checkpoint, "dropout")
        else args_or_checkpoint["dropout"]
    )
    temperature = float(
        args_or_checkpoint.logsumexp_temperature
        if hasattr(args_or_checkpoint, "logsumexp_temperature")
        else args_or_checkpoint["logsumexp_temperature"]
    )
    for item in states:
        state = item["state_dict"] if "state_dict" in item else item
        model = MViTMILHead(
            pooling=pooling,
            dropout=dropout,
            logsumexp_temperature=temperature,
        ).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.inference_mode():
            video_probabilities.append(
                torch.sigmoid(model(tensor)).cpu().numpy()
            )
            clip_probabilities.append(
                torch.sigmoid(model.clip_logits(tensor)).cpu().numpy()
            )
    return (
        np.mean(video_probabilities, axis=0),
        np.mean(clip_probabilities, axis=0),
    )


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


def build_predictions(
    rows: Sequence[baseline.ManifestRow],
    features: np.ndarray,
    starts: np.ndarray,
    checkpoint: dict[str, Any],
    splits: Sequence[str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positions = split_positions(rows)
    predictions: list[dict[str, Any]] = []
    metrics_by_split: dict[str, Any] = {}
    for split in splits:
        index = positions[split]
        probabilities, clip_probabilities = ensemble_probabilities(
            checkpoint["seed_checkpoints"],
            checkpoint["selected_pooling"],
            features[index],
            float(checkpoint["feature_scale"]),
            checkpoint,
            device,
        )
        labels = np.asarray([rows[int(item)].label for item in index], dtype=np.int64)
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
                    "clip_start_frames": starts[int(row_index)].tolist(),
                    "clip_shoplifting_probabilities": clip_probabilities[
                        local_index
                    ].tolist(),
                }
            )
    return predictions, metrics_by_split


def save_predictions(
    output_dir: Path,
    predictions: Sequence[dict[str, Any]],
    included_splits: Sequence[str],
) -> None:
    strict_json_write(
        output_dir / "predictions.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": utc_now(),
            "included_splits": list(included_splits),
            "test_accessed": "test" in included_splits,
            "fixed_threshold": FIXED_THRESHOLD,
            "predictions": list(predictions),
        },
    )


def run_experiment(args: argparse.Namespace) -> None:
    device = baseline.choose_device(args.device)
    data_root = resolve_path(args.data_root)
    manifest_path = resolve_path(args.split_manifest)
    all_rows = baseline.load_manifest(manifest_path, data_root)
    rows = [row for row in all_rows if row.split in {"train", "val"}]
    features, labels, starts, cache_path, cache_reused = ensure_embeddings(
        args, rows, device
    )
    positions = split_positions(rows)
    train_index = positions["train"]
    val_index = positions["val"]
    feature_scale = float(
        np.linalg.norm(features[train_index], axis=-1, keepdims=True).mean()
    )
    if not math.isfinite(feature_scale) or feature_scale <= 0:
        raise RuntimeError(f"Invalid train feature scale: {feature_scale}")
    normalized = features / max(feature_scale, 1e-6)
    tensor = torch.from_numpy(normalized).to(device)
    label_tensor = torch.from_numpy(labels).to(device)
    train_tensor_index = torch.from_numpy(train_index).to(device)
    val_tensor_index = torch.from_numpy(val_index).to(device)
    train_x = tensor.index_select(0, train_tensor_index)
    train_y = label_tensor.index_select(0, train_tensor_index)
    val_x = tensor.index_select(0, val_tensor_index)
    val_y = labels[val_index]

    candidates: list[dict[str, Any]] = []
    for pooling in args.poolings:
        print(f"Training pooling candidate: {pooling}", flush=True)
        seed_checkpoints: list[dict[str, Any]] = []
        for seed in args.seeds:
            state, best_epoch, best_score = train_seed(
                train_x,
                train_y,
                val_x,
                val_y,
                pooling=pooling,
                seed=seed,
                args=args,
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
        val_probabilities, _ = ensemble_probabilities(
            seed_checkpoints,
            pooling,
            features[val_index],
            feature_scale,
            args,
            device,
        )
        val_metrics = baseline.binary_metrics(
            val_y, val_probabilities, FIXED_THRESHOLD
        )
        candidates.append(
            {
                "pooling": pooling,
                "seed_checkpoints": seed_checkpoints,
                "validation_metrics": val_metrics,
                "validation_selection_score": validation_score(val_metrics),
            }
        )

    pooling_order = {name: -index for index, name in enumerate(args.poolings)}
    selected = max(
        candidates,
        key=lambda item: (
            item["validation_selection_score"],
            float(item["validation_metrics"]["balanced_accuracy"]),
            float(item["validation_metrics"]["roc_auc"]),
            pooling_order[item["pooling"]],
        ),
    )
    checkpoint = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "architecture": "MViT-V2-S frozen embeddings + MViTMILHead",
        "manifest_sha256": baseline.sha256_file(manifest_path),
        "test_accessed_during_training": False,
        "selected_pooling": selected["pooling"],
        "feature_scale": feature_scale,
        "clips_per_video": args.clips_per_video,
        "frames_per_clip": FRAMES_PER_CLIP,
        "frame_stride": FRAME_STRIDE,
        "edge_inclusive": True,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "hidden_dimension": HIDDEN_DIMENSION,
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

    predictions, metrics_by_split = build_predictions(
        rows,
        features,
        starts,
        checkpoint,
        ("train", "val"),
        device,
    )
    save_predictions(output_dir, predictions, ("train", "val"))
    summary_candidates = []
    for candidate in candidates:
        summary_candidates.append(
            {
                "pooling": candidate["pooling"],
                "validation_selection_score": candidate[
                    "validation_selection_score"
                ],
                "validation_metrics": candidate["validation_metrics"],
                "seeds": [
                    {
                        "seed": item["seed"],
                        "best_epoch": item["best_epoch"],
                        "best_validation_score": item[
                            "best_validation_score"
                        ],
                    }
                    for item in candidate["seed_checkpoints"]
                ],
            }
        )
    strict_json_write(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": utc_now(),
            "command": "run",
            "test_accessed": False,
            "selection_scope": "validation split only",
            "selected_pooling": selected["pooling"],
            "fixed_threshold": FIXED_THRESHOLD,
            "feature_extractor": "TorchVision MViT-V2-S Kinetics-400",
            "frozen_backbone": True,
            "clips_per_video": args.clips_per_video,
            "edge_inclusive": True,
            "frames_per_clip": FRAMES_PER_CLIP,
            "frame_stride": FRAME_STRIDE,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "feature_scale_source": "train split only",
            "feature_scale": feature_scale,
            "cache": str(cache_path),
            "cache_fully_reused": cache_reused,
            "manifest": str(manifest_path),
            "manifest_sha256": checkpoint["manifest_sha256"],
            "seeds": list(args.seeds),
            "pooling_candidates": summary_candidates,
            "metrics": metrics_by_split,
            "outputs": {
                "checkpoint": str(checkpoint_path),
                "predictions": str(output_dir / "predictions.json"),
            },
        },
    )
    print(
        json.dumps(
            {
                "selected_pooling": selected["pooling"],
                "train": metrics_by_split["train"],
                "val": metrics_by_split["val"],
            },
            indent=2,
        ),
        flush=True,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "seed_checkpoints" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint: {path}")
    if len(checkpoint["seed_checkpoints"]) != 5:
        raise RuntimeError("Checkpoint must contain five seed heads")
    if checkpoint["selected_pooling"] not in POOLING_CHOICES:
        raise RuntimeError("Checkpoint contains an unsupported pooling method")
    if float(checkpoint["threshold"]) != FIXED_THRESHOLD:
        raise RuntimeError("Checkpoint threshold must be fixed at 0.50")
    if int(checkpoint["embedding_dimension"]) != EMBEDDING_DIMENSION:
        raise RuntimeError("Checkpoint embedding dimension is incompatible")
    _ = device
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
    checkpoint = load_checkpoint(checkpoint_path, device)
    if int(checkpoint["clips_per_video"]) != args.clips_per_video:
        raise RuntimeError(
            "--clips-per-video does not match the saved checkpoint"
        )
    if checkpoint["manifest_sha256"] != baseline.sha256_file(manifest_path):
        raise RuntimeError("Checkpoint was trained with a different manifest")
    included_splits = (
        ("train", "val", "test")
        if args.include_test
        else ("train", "val")
    )
    all_rows = baseline.load_manifest(manifest_path, data_root)
    rows = [row for row in all_rows if row.split in included_splits]
    features, _, starts, cache_path, cache_reused = ensure_embeddings(
        args, rows, device
    )
    predictions, metrics_by_split = build_predictions(
        rows,
        features,
        starts,
        checkpoint,
        included_splits,
        device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(output_dir, predictions, included_splits)
    strict_json_write(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": utc_now(),
            "command": "evaluate",
            "test_accessed": args.include_test,
            "included_splits": list(included_splits),
            "selected_pooling": checkpoint["selected_pooling"],
            "selection_scope": "validation split only during training",
            "fixed_threshold": FIXED_THRESHOLD,
            "feature_extractor": "TorchVision MViT-V2-S Kinetics-400",
            "clips_per_video": args.clips_per_video,
            "cache": str(cache_path),
            "cache_fully_reused": cache_reused,
            "checkpoint": str(checkpoint_path),
            "metrics": metrics_by_split,
        },
    )
    print(json.dumps(metrics_by_split, indent=2), flush=True)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Defaults to data/cache/mvit-v2-s-dense-N-embeddings.npz.",
    )
    parser.add_argument(
        "--clips-per-video", type=int, choices=(15, 21), default=15
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights-file", type=Path, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=2)
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Ignore compatible cached MViT embeddings for included splits.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen MViT-V2-S dense-clip MIL experiment with sealed test access."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Extract/reuse train+val embeddings and train all candidates."
    )
    add_common_arguments(run_parser)
    run_parser.add_argument(
        "--poolings",
        nargs="+",
        choices=POOLING_CHOICES,
        default=list(POOLING_CHOICES),
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
        help="Evaluate train+val; add --include-test to open the test split.",
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
    if args.command == "run":
        if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
            raise ValueError("--seeds must contain exactly five distinct values")
        if len(set(args.poolings)) != len(args.poolings):
            raise ValueError("--poolings must not contain duplicates")
        if args.max_epochs <= 0 or args.eval_every <= 0 or args.patience <= 0:
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
