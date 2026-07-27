"""Fine-tune a Kinetics video backbone with weak video-level shoplifting labels.

The reviewed actor/clothing manifest is the only split source. ``train`` opens
videos assigned to train and validation; it never opens test, OOD, qualitative,
or excluded videos. The test split is opened only by the explicit command:

    python scripts/shoplifting_video_finetune.py evaluate --include-test

Default training uses TorchVision MViT-V2-S Kinetics-400, seven temporally
jittered clips per training video, fifteen fixed dense clips per validation
video, actor-group-balanced sampling, BF16 autocast, gradient accumulation,
and a validation-only early-stopping score:

    validation balanced accuracy + 0.05 * validation ROC-AUC

Each video is a weakly labelled bag. The model predicts one logit per clip and
combines clip logits with top-1 or normalized log-sum-exp MIL pooling. Spatial
augmentation parameters are shared by every frame and clip in one video bag,
so augmentation cannot create frame-to-frame flicker.

Examples from the repository root:

    .venv-yolo\\Scripts\\python.exe scripts\\shoplifting_video_finetune.py train

    .venv-yolo\\Scripts\\python.exe scripts\\shoplifting_video_finetune.py train ^
      --backbone r3d_18 --unfreeze full --train-clips 7

    .venv-yolo\\Scripts\\python.exe scripts\\shoplifting_video_finetune.py ^
      evaluate --include-test

The default output directory is ``output/shoplifting-video-finetune``. It is
separate from the published frozen-R3D benchmark and never writes below
``docs/results``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.video import (
    MViT_V2_S_Weights,
    R3D_18_Weights,
    mvit_v2_s,
    r3d_18,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import shoplifting_mil_baseline as baseline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "shoplifting-video-dataset"
DEFAULT_MANIFEST = (
    ROOT / "docs" / "results" / "shoplifting" / "actor-clothing-disjoint-split.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "output" / "shoplifting-video-finetune"
DEFAULT_MVIT_WEIGHTS = (
    ROOT / "data" / "torch-cache" / "hub" / "checkpoints" / "mvit_v2_s-ae3be167.pth"
)
DEFAULT_R3D_WEIGHTS = (
    ROOT / "data" / "torch-cache" / "hub" / "checkpoints" / "r3d_18-b3b3357e.pth"
)

SCRIPT_VERSION = "1.0.0"
LABEL_NAMES = {0: "Normal", 1: "Shoplifting"}
TRAINABLE_SPLITS = frozenset({"train", "val"})
MIL_CHOICES = ("top1", "logsumexp")
BACKBONE_CHOICES = ("mvit_v2_s", "r3d_18")
UNFREEZE_CHOICES = ("head", "last2", "full")
FIXED_THRESHOLD = 0.50

if (ROOT / "data" / "torch-cache").is_dir():
    os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "torch-cache"))


@dataclass(frozen=True)
class VideoRow:
    relative_path: str
    split: str
    actor_group: str
    reason: str
    label: int


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    crop_size: int
    resize_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    weight_name: str
    weight_url: str


BACKBONE_SPECS = {
    "mvit_v2_s": BackboneSpec(
        name="mvit_v2_s",
        crop_size=224,
        resize_size=(256, 256),
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
        weight_name="MViT_V2_S_Weights.KINETICS400_V1",
        weight_url=MViT_V2_S_Weights.KINETICS400_V1.url,
    ),
    "r3d_18": BackboneSpec(
        name="r3d_18",
        crop_size=112,
        resize_size=(128, 171),
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        weight_name="R3D_18_Weights.KINETICS400_V1",
        weight_url=R3D_18_Weights.KINETICS400_V1.url,
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def natural_key(text: str) -> tuple[tuple[int, Any], ...]:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"([0-9]+)", text)
    )


def label_from_path(relative_path: str) -> int:
    normalized = relative_path.replace("\\", "/").casefold()
    if normalized.startswith("normal/"):
        return 0
    if normalized.startswith("shoplifting/"):
        return 1
    raise ValueError(f"Cannot infer class from manifest path: {relative_path}")


def load_fixed_manifest(
    manifest_path: Path,
    data_root: Path,
    allowed_splits: set[str] | frozenset[str],
) -> tuple[list[VideoRow], str]:
    """Load assignments, but only resolve/open source paths in allowed splits."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Reviewed split manifest is missing: {manifest_path}")
    rows: list[VideoRow] = []
    seen_paths: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "split", "actor_group", "reason"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Manifest requires columns {sorted(required)}; got {reader.fieldnames}"
            )
        for raw in reader:
            split = raw["split"].strip().casefold()
            if split not in allowed_splits:
                # Deliberately do not resolve, stat, hash, or decode sealed video rows.
                continue
            relative_path = Path(raw["relative_path"].strip()).as_posix()
            key = relative_path.casefold()
            if key in seen_paths:
                raise ValueError(f"Duplicate allowed manifest path: {relative_path}")
            seen_paths.add(key)
            source_path = data_root / Path(relative_path)
            if not source_path.is_file():
                raise FileNotFoundError(f"Manifest source is missing: {source_path}")
            actor_group = raw["actor_group"].strip()
            if not actor_group:
                raise ValueError(f"Missing actor_group for {relative_path}")
            rows.append(
                VideoRow(
                    relative_path=relative_path,
                    split=split,
                    actor_group=actor_group,
                    reason=raw["reason"].strip(),
                    label=label_from_path(relative_path),
                )
            )
    rows.sort(key=lambda row: natural_key(row.relative_path))
    for split in allowed_splits:
        split_rows = [row for row in rows if row.split == split]
        if not split_rows:
            raise RuntimeError(f"Manifest has no accessible {split} rows")
        if {row.label for row in split_rows} != {0, 1}:
            raise RuntimeError(f"Accessible {split} split must contain both classes")
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(row.actor_group.casefold(), set()).add(row.split)
    overlaps = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) > 1
    }
    if overlaps:
        raise RuntimeError(f"Actor groups cross accessible splits: {overlaps}")
    return rows, sha256_file(manifest_path)


def balanced_limit(rows: Sequence[VideoRow], limit: int) -> list[VideoRow]:
    """Small deterministic subset for smoke tests; zero keeps every row."""

    if limit <= 0 or limit >= len(rows):
        return list(rows)
    by_label = {
        label: sorted(
            (row for row in rows if row.label == label),
            key=lambda row: natural_key(row.relative_path),
        )
        for label in (0, 1)
    }
    selected: list[VideoRow] = []
    cursor = {0: 0, 1: 0}
    next_label = 0
    while len(selected) < limit:
        candidates = [next_label, 1 - next_label]
        chosen: int | None = None
        for label in candidates:
            if cursor[label] < len(by_label[label]):
                chosen = label
                break
        if chosen is None:
            break
        selected.append(by_label[chosen][cursor[chosen]])
        cursor[chosen] += 1
        next_label = 1 - chosen
    if {row.label for row in selected} != {0, 1}:
        raise RuntimeError("A limited split must retain both classes")
    return sorted(selected, key=lambda row: natural_key(row.relative_path))


def dense_starts(frame_count: int, clip_length: int, frame_stride: int, count: int) -> list[int]:
    span = (clip_length - 1) * frame_stride + 1
    maximum = max(0, frame_count - span)
    if count == 1:
        return [maximum // 2]
    return np.rint(np.linspace(0, maximum, count)).astype(np.int64).tolist()


def jittered_starts(
    frame_count: int,
    clip_length: int,
    frame_stride: int,
    count: int,
    rng: random.Random,
) -> list[int]:
    """Stratified temporal jitter covers the video without fixed clip centers."""

    span = (clip_length - 1) * frame_stride + 1
    maximum = max(0, frame_count - span)
    if maximum == 0:
        return [0] * count
    edges = np.linspace(0.0, float(maximum + 1), count + 1)
    starts: list[int] = []
    for index in range(count):
        low = min(maximum, int(math.floor(edges[index])))
        high = min(maximum, max(low, int(math.ceil(edges[index + 1])) - 1))
        starts.append(rng.randint(low, high))
    return starts


def decode_clips(
    path: Path,
    starts: Sequence[int],
    clip_length: int,
    frame_stride: int,
) -> np.ndarray:
    indices_by_clip = [
        [start + offset * frame_stride for offset in range(clip_length)]
        for start in starts
    ]
    needed = sorted({index for indices in indices_by_clip for index in indices})
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    decoded: dict[int, np.ndarray] = {}
    try:
        for frame_index in range(needed[-1] + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not decode frame {frame_index} from {path}")
            if frame_index in needed:
                decoded[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    clips = [
        np.stack([decoded[index] for index in indices], axis=0)
        for indices in indices_by_clip
    ]
    return np.stack(clips, axis=0)


def random_resized_crop_parameters(
    height: int,
    width: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    area = height * width
    for _ in range(10):
        target_area = area * rng.uniform(0.72, 1.0)
        aspect = math.exp(rng.uniform(math.log(0.85), math.log(1.18)))
        crop_width = int(round(math.sqrt(target_area * aspect)))
        crop_height = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = rng.randint(0, height - crop_height)
            left = rng.randint(0, width - crop_width)
            return top, left, crop_height, crop_width
    side = min(height, width)
    return (height - side) // 2, (width - side) // 2, side, side


def augment_video_bag(
    clips: np.ndarray,
    spec: BackboneSpec,
    rng: random.Random,
) -> torch.Tensor:
    """Apply one spatial/color transform consistently to every bag frame."""

    bag_count, clip_length, height, width, _ = clips.shape
    tensor = torch.from_numpy(np.ascontiguousarray(clips)).permute(0, 1, 4, 2, 3)
    flat = tensor.reshape(bag_count * clip_length, 3, height, width)
    top, left, crop_height, crop_width = random_resized_crop_parameters(
        height, width, rng
    )
    flat = TF.resized_crop(
        flat,
        top=top,
        left=left,
        height=crop_height,
        width=crop_width,
        size=[spec.crop_size, spec.crop_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=False,
    )
    if rng.random() < 0.5:
        flat = TF.hflip(flat)
    flat = TF.convert_image_dtype(flat, torch.float32)
    # The factors are shared across frames; augmentation cannot introduce flicker.
    flat = TF.adjust_brightness(flat, rng.uniform(0.90, 1.10))
    flat = TF.adjust_contrast(flat, rng.uniform(0.90, 1.10))
    flat = TF.adjust_saturation(flat, rng.uniform(0.90, 1.10))
    flat = TF.normalize(flat, mean=list(spec.mean), std=list(spec.std))
    return (
        flat.reshape(bag_count, clip_length, 3, spec.crop_size, spec.crop_size)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def deterministic_video_bag(
    clips: np.ndarray,
    preprocess: nn.Module,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(clips)).permute(0, 1, 4, 2, 3)
    # TorchVision VideoClassification accepts (B, T, C, H, W) and returns
    # (B, C, T, H, W).
    return preprocess(tensor)


class VideoBagDataset(Dataset[dict[str, Any]]):
    """One item is one source video containing multiple temporal clips."""

    def __init__(
        self,
        rows: Sequence[VideoRow],
        data_root: Path,
        backbone: str,
        clip_length: int,
        frame_stride: int,
        clips_per_video: int,
        seed: int,
        training: bool,
    ) -> None:
        self.rows = list(rows)
        self.data_root = data_root
        self.spec = BACKBONE_SPECS[backbone]
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.clips_per_video = clips_per_video
        self.seed = seed
        self.training = training
        self.epoch = 0
        weights = (
            MViT_V2_S_Weights.KINETICS400_V1
            if backbone == "mvit_v2_s"
            else R3D_18_Weights.KINETICS400_V1
        )
        self.preprocess = weights.transforms()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.data_root / Path(row.relative_path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open {path}")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
        if frame_count <= 0:
            raise RuntimeError(f"Invalid frame count for {path}")
        rng = random.Random(
            stable_seed(
                f"{self.seed}|{self.epoch if self.training else 0}|"
                f"{row.relative_path}|{row.actor_group}"
            )
        )
        if self.training:
            starts = jittered_starts(
                frame_count,
                self.clip_length,
                self.frame_stride,
                self.clips_per_video,
                rng,
            )
        else:
            starts = dense_starts(
                frame_count,
                self.clip_length,
                self.frame_stride,
                self.clips_per_video,
            )
        clips = decode_clips(path, starts, self.clip_length, self.frame_stride)
        video = (
            augment_video_bag(clips, self.spec, rng)
            if self.training
            else deterministic_video_bag(clips, self.preprocess)
        )
        return {
            "video": video,
            "label": row.label,
            "relative_path": row.relative_path,
            "split": row.split,
            "actor_group": row.actor_group,
            "clip_starts": torch.tensor(starts, dtype=torch.int64),
        }


def load_official_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported official weights file: {path}")
    return {
        str(key).removeprefix("module."): value
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
    }


class VideoMILModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pooling: str,
        unfreeze: str,
        logsumexp_temperature: float,
        pretrained: bool,
        weights_file: Path | None,
    ) -> None:
        super().__init__()
        if pooling not in MIL_CHOICES:
            raise ValueError(f"Unsupported MIL pooling: {pooling}")
        self.backbone_name = backbone_name
        self.pooling = pooling
        self.unfreeze_policy = unfreeze
        self.logsumexp_temperature = float(logsumexp_temperature)

        if backbone_name == "mvit_v2_s":
            weights = MViT_V2_S_Weights.KINETICS400_V1
            if pretrained and weights_file is None:
                self.backbone = mvit_v2_s(weights=weights, progress=True)
            else:
                self.backbone = mvit_v2_s(weights=None)
                if pretrained and weights_file is not None:
                    self.backbone.load_state_dict(
                        load_official_state(weights_file), strict=True
                    )
            in_features = self.backbone.head[-1].in_features
            self.backbone.head[-1] = nn.Linear(in_features, 1)
        elif backbone_name == "r3d_18":
            weights = R3D_18_Weights.KINETICS400_V1
            if pretrained and weights_file is None:
                self.backbone = r3d_18(weights=weights, progress=True)
            else:
                self.backbone = r3d_18(weights=None)
                if pretrained and weights_file is not None:
                    self.backbone.load_state_dict(
                        load_official_state(weights_file), strict=True
                    )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, 1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        self.configure_trainable(unfreeze)

    def head_module(self) -> nn.Module:
        return self.backbone.head if self.backbone_name == "mvit_v2_s" else self.backbone.fc

    def configure_trainable(self, policy: str) -> None:
        if policy not in UNFREEZE_CHOICES:
            raise ValueError(f"Unsupported unfreeze policy: {policy}")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if policy == "full":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
        elif policy == "head":
            for parameter in self.head_module().parameters():
                parameter.requires_grad = True
        elif self.backbone_name == "mvit_v2_s":
            for module in (
                *self.backbone.blocks[-2:],
                self.backbone.norm,
                self.backbone.head,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        else:
            # R3D has four residual stages; "last2" maps to layer3+layer4.
            for module in (
                self.backbone.layer3,
                self.backbone.layer4,
                self.backbone.fc,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def set_training_mode(self) -> None:
        if self.unfreeze_policy == "full":
            self.backbone.train()
            return
        # Frozen Dropout/BatchNorm/stochastic-depth layers remain deterministic.
        self.backbone.eval()
        if self.unfreeze_policy == "head":
            self.head_module().train()
        elif self.backbone_name == "mvit_v2_s":
            self.backbone.blocks[-2:].train()
            self.backbone.norm.train()
            self.backbone.head.train()
        else:
            self.backbone.layer3.train()
            self.backbone.layer4.train()
            self.backbone.fc.train()

    def clip_logits(self, clips: torch.Tensor) -> torch.Tensor:
        return self.backbone(clips).reshape(-1)

    def pool_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.pooling == "top1":
            return logits.max(dim=1).values
        temperature = self.logsumexp_temperature
        return temperature * (
            torch.logsumexp(logits / temperature, dim=1)
            - math.log(logits.shape[1])
        )

    def forward(self, bags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, clips, channels, frames, height, width = bags.shape
        flat = bags.reshape(batch * clips, channels, frames, height, width)
        clip_logits = self.clip_logits(flat).reshape(batch, clips)
        return self.pool_logits(clip_logits), clip_logits


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


def amp_settings(
    requested: str,
    device: torch.device,
) -> tuple[torch.dtype | None, str]:
    if device.type != "cuda" or requested == "fp32":
        return None, "fp32"
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this GPU does not support it")
        return torch.bfloat16, "bf16"
    if requested == "fp16":
        return torch.float16, "fp16"
    raise ValueError(f"Unsupported precision: {requested}")


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def actor_balanced_sampler(
    rows: Sequence[VideoRow],
    seed: int,
) -> WeightedRandomSampler:
    counts = Counter(row.actor_group.casefold() for row in rows)
    weights = torch.tensor(
        [1.0 / counts[row.actor_group.casefold()] for row in rows],
        dtype=torch.double,
    )
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=len(rows),
        replacement=True,
        generator=generator,
    )


def make_train_loader(
    dataset: VideoBagDataset,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> DataLoader:
    dataset.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=actor_balanced_sampler(dataset.rows, args.seed + epoch),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init,
    )


def make_eval_loader(
    dataset: VideoBagDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init,
    )


def optimizer_for_model(
    model: VideoMILModel,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head_ids = {id(parameter) for parameter in model.head_module().parameters()}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in head_ids
    ]
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    groups: list[dict[str, Any]] = []
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_lr,
                "name": "backbone",
            }
        )
    if head_parameters:
        groups.append({"params": head_parameters, "lr": head_lr, "name": "head"})
    if not groups:
        raise RuntimeError("No trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def class_positive_weight(rows: Sequence[VideoRow], device: torch.device) -> torch.Tensor:
    negatives = sum(row.label == 0 for row in rows)
    positives = sum(row.label == 1 for row in rows)
    if not negatives or not positives:
        raise RuntimeError("Training split requires both classes")
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def train_one_epoch(
    model: VideoMILModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler,
    grad_accum: int,
    grad_clip: float,
) -> dict[str, float]:
    model.set_training_mode()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_videos = 0
    optimizer_steps = 0
    for batch_index, batch in enumerate(loader):
        bags = batch["video"].to(device, non_blocking=True)
        labels = batch["label"].to(device, dtype=torch.float32, non_blocking=True)
        with autocast_context(device, amp_dtype):
            video_logits, _ = model(bags)
            unscaled_loss = criterion(video_logits, labels)
            loss = unscaled_loss / grad_accum
        scaler.scale(loss).backward()
        should_step = (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        batch_size = int(labels.shape[0])
        total_loss += float(unscaled_loss.detach().cpu()) * batch_size
        total_videos += batch_size
    return {
        "loss": total_loss / max(1, total_videos),
        "videos": float(total_videos),
        "optimizer_steps": float(optimizer_steps),
    }


@torch.inference_mode()
def predict_dataset(
    model: VideoMILModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    clip_batch_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        # Evaluation loader is deliberately one source video per batch.
        bag = batch["video"][0]
        clip_logits_parts: list[torch.Tensor] = []
        for offset in range(0, bag.shape[0], clip_batch_size):
            clips = bag[offset : offset + clip_batch_size].to(
                device, non_blocking=True
            )
            with autocast_context(device, amp_dtype):
                logits = model.clip_logits(clips)
            clip_logits_parts.append(logits.float().cpu())
        clip_logits = torch.cat(clip_logits_parts).unsqueeze(0)
        video_logit = model.pool_logits(clip_logits).item()
        probability = float(torch.sigmoid(torch.tensor(video_logit)).item())
        truth = int(batch["label"].item())
        predicted = int(probability >= FIXED_THRESHOLD)
        starts = batch["clip_starts"][0].tolist()
        predictions.append(
            {
                "relative_path": batch["relative_path"][0],
                "split": batch["split"][0],
                "actor_group": batch["actor_group"][0],
                "ground_truth": truth,
                "ground_truth_name": LABEL_NAMES[truth],
                "shoplifting_probability": probability,
                "predicted_label": predicted,
                "predicted_name": LABEL_NAMES[predicted],
                "correct": predicted == truth,
                "video_logit": video_logit,
                "clip_logits": clip_logits.squeeze(0).tolist(),
                "clip_start_frames": starts,
            }
        )
    return predictions


def metrics_for_predictions(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = baseline.binary_metrics(
        np.asarray([row["ground_truth"] for row in predictions], dtype=np.int64),
        np.asarray(
            [row["shoplifting_probability"] for row in predictions],
            dtype=np.float64,
        ),
        threshold=FIXED_THRESHOLD,
    )
    actor_rows: list[dict[str, Any]] = []
    for actor_group in sorted({row["actor_group"] for row in predictions}):
        members = [row for row in predictions if row["actor_group"] == actor_group]
        actor_metrics = baseline.binary_metrics(
            np.asarray([row["ground_truth"] for row in members], dtype=np.int64),
            np.asarray(
                [row["shoplifting_probability"] for row in members],
                dtype=np.float64,
            ),
            threshold=FIXED_THRESHOLD,
        )
        actor_rows.append(
            {
                "actor_group": actor_group,
                "support": actor_metrics["support"],
                "accuracy": actor_metrics["accuracy"],
                "balanced_accuracy": actor_metrics["balanced_accuracy"],
                "recall": actor_metrics["recall"],
                "specificity": actor_metrics["specificity"],
            }
        )
    metrics["actor_groups"] = actor_rows
    metrics["actor_group_count"] = len(actor_rows)
    return metrics


def validation_selection_score(metrics: dict[str, Any]) -> float:
    auc = metrics["roc_auc"]
    if auc is None:
        return -math.inf
    return float(metrics["balanced_accuracy"]) + 0.05 * float(auc)


def save_checkpoint(
    path: Path,
    model: VideoMILModel,
    config: dict[str, Any],
    epoch: int,
    validation_metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "epoch": epoch,
        "validation_selection_score": validation_selection_score(validation_metrics),
        "validation_metrics": baseline.json_ready(validation_metrics),
        "config": baseline.json_ready(config),
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    baseline.write_json(
        path.with_suffix(".json"),
        {
            key: value
            for key, value in payload.items()
            if key != "model_state_dict"
        }
        | {"checkpoint_bytes": path.stat().st_size},
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[VideoMILModel, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"Unsupported checkpoint: {path}")
    config = payload["config"]
    model = VideoMILModel(
        backbone_name=config["backbone"],
        pooling=config["pooling"],
        unfreeze=config["unfreeze"],
        logsumexp_temperature=float(config["logsumexp_temperature"]),
        pretrained=False,
        weights_file=None,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def environment_info(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torchvision": __import__("torchvision").__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "bf16_supported": (
            torch.cuda.is_bf16_supported() if device.type == "cuda" else False
        ),
    }


def experiment_config(
    args: argparse.Namespace,
    data_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    train_rows: Sequence[VideoRow],
    val_rows: Sequence[VideoRow],
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    spec = BACKBONE_SPECS[args.backbone]
    weights_file = resolve_weights_file(args.backbone, args.weights_file)
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "opened_video_splits_during_training": ["train", "val"],
        "sealed_video_splits_during_training": [
            "test",
            "ood",
            "qualitative",
            "excluded",
        ],
        "train_videos": len(train_rows),
        "val_videos": len(val_rows),
        "train_actor_groups": sorted({row.actor_group for row in train_rows}),
        "val_actor_groups": sorted({row.actor_group for row in val_rows}),
        "limit_train_videos": args.limit_train_videos,
        "limit_val_videos": args.limit_val_videos,
        "backbone": args.backbone,
        "pretrained_weights": spec.weight_name,
        "weights_url": spec.weight_url,
        "weights_file": str(weights_file) if weights_file else None,
        "unfreeze": args.unfreeze,
        "pooling": args.pooling,
        "logsumexp_temperature": args.logsumexp_temperature,
        "clip_length": args.clip_length,
        "frame_stride": args.frame_stride,
        "train_clips": args.train_clips,
        "eval_clips": args.eval_clips,
        "crop_size": spec.crop_size,
        "train_temporal_sampling": "stratified random jitter per epoch",
        "validation_temporal_sampling": "fixed edge-inclusive dense clips",
        "consistent_spatial_augmentation": True,
        "actor_group_balanced_sampling": True,
        "batch_size_videos": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "precision": precision,
        "backbone_learning_rate": args.backbone_lr,
        "head_learning_rate": args.head_lr,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.grad_clip,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "seed": args.seed,
        "fixed_threshold": FIXED_THRESHOLD,
        "checkpoint_selection": (
            "validation balanced_accuracy + 0.05 * validation ROC-AUC"
        ),
        "test_access_policy": "evaluate --include-test only",
        "environment": environment_info(device),
    }


def resolve_weights_file(backbone: str, requested: Path | None) -> Path | None:
    if requested is not None:
        path = resolve_path(requested)
        if not path.is_file():
            raise FileNotFoundError(f"Official weights file is missing: {path}")
        return path
    default = DEFAULT_MVIT_WEIGHTS if backbone == "mvit_v2_s" else DEFAULT_R3D_WEIGHTS
    return default if default.is_file() else None


def output_predictions(
    output_dir: Path,
    stem: str,
    predictions: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    baseline.write_json(
        output_dir / f"{stem}.json",
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "metrics": metrics,
            "predictions": list(predictions),
        },
    )
    rows = []
    for prediction in predictions:
        row = prediction.copy()
        row["clip_logits"] = "|".join(
            f"{float(value):.9g}" for value in prediction["clip_logits"]
        )
        row["clip_start_frames"] = "|".join(
            str(value) for value in prediction["clip_start_frames"]
        )
        rows.append(row)
    baseline.write_csv(
        output_dir / f"{stem}.csv",
        rows,
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
            "video_logit",
            "clip_logits",
            "clip_start_frames",
        ),
    )


def print_metrics(split: str, metrics: dict[str, Any]) -> None:
    auc = metrics["roc_auc"]
    auc_text = "n/a" if auc is None else f"{auc:.3f}"
    print(
        f"{split}: n={metrics['support']}, accuracy={metrics['accuracy']:.3f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
        f"F1={metrics['f1']:.3f}, ROC-AUC={auc_text}",
        flush=True,
    )


def run_train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    data_root = resolve_path(args.data_root)
    manifest_path = resolve_path(args.split_manifest)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest_sha256 = load_fixed_manifest(
        manifest_path, data_root, TRAINABLE_SPLITS
    )
    train_rows = balanced_limit(
        [row for row in rows if row.split == "train"], args.limit_train_videos
    )
    val_rows = balanced_limit(
        [row for row in rows if row.split == "val"], args.limit_val_videos
    )
    device = choose_device(args.device)
    amp_dtype, precision = amp_settings(args.precision, device)
    print(f"Device: {device}; precision: {precision}", flush=True)

    train_dataset = VideoBagDataset(
        train_rows,
        data_root,
        args.backbone,
        args.clip_length,
        args.frame_stride,
        args.train_clips,
        args.seed,
        training=True,
    )
    val_dataset = VideoBagDataset(
        val_rows,
        data_root,
        args.backbone,
        args.clip_length,
        args.frame_stride,
        args.eval_clips,
        args.seed,
        training=False,
    )
    val_loader = make_eval_loader(val_dataset, args, device)
    weights_file = resolve_weights_file(args.backbone, args.weights_file)
    model = VideoMILModel(
        args.backbone,
        args.pooling,
        args.unfreeze,
        args.logsumexp_temperature,
        pretrained=True,
        weights_file=weights_file,
    ).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Backbone: {args.backbone}; trainable parameters: "
        f"{trainable_parameters:,}/{total_parameters:,}",
        flush=True,
    )
    optimizer = optimizer_for_model(
        model, args.backbone_lr, args.head_lr, args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.max_epochs)
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=class_positive_weight(train_rows, device)
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and amp_dtype == torch.float16,
    )
    config = experiment_config(
        args,
        data_root,
        manifest_path,
        manifest_sha256,
        train_rows,
        val_rows,
        device,
        precision,
    )
    config["trainable_parameters"] = trainable_parameters
    config["total_parameters"] = total_parameters
    baseline.write_json(output_dir / "config.json", config)
    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint is not None
        else output_dir / "checkpoint.pt"
    )

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    stale_epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        started = time.perf_counter()
        train_loader = make_train_loader(train_dataset, args, device, epoch)
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            amp_dtype,
            scaler,
            args.grad_accum,
            args.grad_clip,
        )
        val_predictions = predict_dataset(
            model,
            val_loader,
            device,
            amp_dtype,
            args.eval_clip_batch_size,
        )
        val_metrics = metrics_for_predictions(val_predictions)
        score = validation_selection_score(val_metrics)
        improved = score > best_score + args.min_delta
        if improved:
            best_score = score
            stale_epochs = 0
            save_checkpoint(checkpoint_path, model, config, epoch, val_metrics)
            output_predictions(
                output_dir,
                "best-val-predictions",
                val_predictions,
                val_metrics,
            )
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_videos": int(train_result["videos"]),
            "optimizer_steps": int(train_result["optimizer_steps"]),
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "validation_selection_score": score,
            "selected": improved,
            "stale_epochs": stale_epochs,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        baseline.write_json(output_dir / "history.json", history)
        csv_rows = []
        for item in history:
            csv_row = item.copy()
            csv_row["learning_rates"] = "|".join(
                f"{float(value):.9g}" for value in item["learning_rates"]
            )
            csv_rows.append(csv_row)
        baseline.write_csv(
            output_dir / "history.csv",
            csv_rows,
            tuple(csv_rows[0]),
        )
        print(
            f"Epoch {epoch:02d}/{args.max_epochs}: "
            f"loss={train_result['loss']:.4f}, "
            f"val_BA={val_metrics['balanced_accuracy']:.3f}, "
            f"val_AUC={val_metrics['roc_auc']:.3f}, "
            f"{'saved' if improved else f'patience {stale_epochs}/{args.patience}'}, "
            f"{row['seconds']:.1f}s",
            flush=True,
        )
        scheduler.step()
        if stale_epochs >= args.patience:
            print(f"Early stopping after epoch {epoch}", flush=True)
            break

    if not checkpoint_path.is_file():
        raise RuntimeError("Training completed without a checkpoint")
    print(
        f"Training complete. Best checkpoint: {checkpoint_path}\n"
        "The test split remained sealed. Run evaluate --include-test explicitly.",
        flush=True,
    )


def evaluate_rows(
    model: VideoMILModel,
    rows: Sequence[VideoRow],
    data_root: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_by_split: dict[str, Any] = {}
    all_predictions: list[dict[str, Any]] = []
    for split in sorted({row.split for row in rows}):
        split_rows = [row for row in rows if row.split == split]
        dataset = VideoBagDataset(
            split_rows,
            data_root,
            config["backbone"],
            int(config["clip_length"]),
            int(config["frame_stride"]),
            int(config["eval_clips"]),
            int(config["seed"]),
            training=False,
        )
        loader = make_eval_loader(dataset, args, device)
        predictions = predict_dataset(
            model,
            loader,
            device,
            amp_dtype,
            args.eval_clip_batch_size,
        )
        metrics = metrics_for_predictions(predictions)
        metrics_by_split[split] = metrics
        all_predictions.extend(predictions)
        print_metrics(split, metrics)
    return metrics_by_split, all_predictions


def run_evaluate(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    data_root = resolve_path(args.data_root)
    manifest_path = resolve_path(args.split_manifest)
    output_dir = resolve_path(args.output_dir)
    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint is not None
        else output_dir / "checkpoint.pt"
    )
    device = choose_device(args.device)
    model, payload = load_checkpoint(checkpoint_path, device)
    config = payload["config"]
    if config["manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError("Checkpoint manifest SHA-256 does not match the fixed manifest")
    allowed = {"val", "test"} if args.include_test else {"val"}
    rows, _ = load_fixed_manifest(manifest_path, data_root, allowed)
    _, precision = amp_settings(args.precision, device)
    amp_dtype, _ = amp_settings(precision, device)
    metrics, predictions = evaluate_rows(
        model, rows, data_root, config, args, device, amp_dtype
    )
    suffix = "with-test" if args.include_test else "val-only"
    payload_out = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload["epoch"],
        "manifest_sha256": config["manifest_sha256"],
        "included_splits": sorted(allowed),
        "test_opened": bool(args.include_test),
        "metrics": metrics,
    }
    baseline.write_json(output_dir / f"evaluation-{suffix}.json", payload_out)
    output_predictions(
        output_dir,
        f"predictions-{suffix}",
        predictions,
        {"splits": metrics},
    )
    print(f"Evaluation written to {output_dir}; test_opened={args.include_test}")


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to OUTPUT_DIR/checkpoint.pt.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-clip-batch-size", type=int, default=3)
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=20260727)


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backbone", choices=BACKBONE_CHOICES, default="mvit_v2_s")
    parser.add_argument("--unfreeze", choices=UNFREEZE_CHOICES, default="last2")
    parser.add_argument("--pooling", choices=MIL_CHOICES, default="logsumexp")
    parser.add_argument("--logsumexp-temperature", type=float, default=0.5)
    parser.add_argument("--weights-file", type=Path, default=None)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--train-clips", type=int, default=7)
    parser.add_argument("--eval-clips", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--limit-train-videos",
        type=int,
        default=0,
        help="Smoke-test only. Zero uses the complete train split.",
    )
    parser.add_argument(
        "--limit-val-videos",
        type=int,
        default=0,
        help="Smoke-test only. Zero uses the complete validation split.",
    )


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "num_workers",
        "limit_train_videos",
        "limit_val_videos",
    )
    for name in positive:
        if getattr(args, name, 0) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    for name in (
        "eval_clip_batch_size",
        "clip_length",
        "frame_stride",
        "train_clips",
        "eval_clips",
        "batch_size",
        "grad_accum",
        "max_epochs",
        "patience",
    ):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "logsumexp_temperature") and args.logsumexp_temperature <= 0:
        raise ValueError("--logsumexp-temperature must be positive")
    if hasattr(args, "grad_clip") and args.grad_clip < 0:
        raise ValueError("--grad-clip cannot be negative")
    for name in ("limit_train_videos", "limit_val_videos"):
        if hasattr(args, name) and 0 < getattr(args, name) < 2:
            raise ValueError(
                f"--{name.replace('_', '-')} must be zero or at least two "
                "so both classes remain represented"
            )
    if (
        getattr(args, "backbone", None) == "mvit_v2_s"
        and getattr(args, "unfreeze", None) == "full"
        and args.batch_size * args.train_clips > 3
    ):
        raise ValueError(
            "Full MViT fine-tuning on the 8 GiB GPU is limited to at most "
            "three clips per optimizer forward; use --train-clips 3 --batch-size 1."
        )
    if (
        getattr(args, "backbone", None) == "mvit_v2_s"
        and getattr(args, "unfreeze", None) == "last2"
        and args.batch_size * args.train_clips > 7
    ):
        raise ValueError(
            "MViT last2 fine-tuning is limited to seven clips per forward on this GPU."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Weak-label MViT/R3D video-bag fine-tuning with validation-only "
            "selection and an explicitly sealed test split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train with train+val videos only; never opens test videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_shared_arguments(train_parser)
    add_train_arguments(train_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate validation; --include-test explicitly opens the fixed test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_shared_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly open and score the fixed test split.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        parser.error(f"Unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
