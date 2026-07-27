"""Build the fixed recall-first OR ensemble benchmark artifacts.

The final risk for each source video is the elementwise maximum of:

* the frozen R3D-18 MIL video probability; and
* the YOLO26-guided dual-view MViT MIL video probability.

This is a fixed CCTV recall-first OR rule:

    final_risk = max(r3d18_probability, yolo_mvit_probability)
    prediction = final_risk >= 0.50

The script does not tune weights or thresholds. It creates the English
benchmark package below ``docs/results/shoplifting-yolo26-v2`` and explicitly
marks the opened test result as a development benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

import shoplifting_mil_baseline as baseline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_R3D_PREDICTIONS = (
    ROOT / "docs" / "results" / "shoplifting" / "predictions.json"
)
DEFAULT_R3D_METRICS = (
    ROOT / "docs" / "results" / "shoplifting" / "metrics.json"
)
DEFAULT_R3D_CHECKPOINT = (
    ROOT / "docs" / "results" / "shoplifting" / "checkpoint.pt"
)
DEFAULT_YOLO_PREDICTIONS = (
    ROOT
    / "docs"
    / "results"
    / "shoplifting-yolo26-v2"
    / "yolo-mvit-predictions.json"
)
DEFAULT_YOLO_SUMMARY = (
    ROOT
    / "docs"
    / "results"
    / "shoplifting-yolo26-v2"
    / "yolo-mvit-summary.json"
)
DEFAULT_YOLO_CHECKPOINT = (
    ROOT
    / "docs"
    / "results"
    / "shoplifting-yolo26-v2"
    / "yolo-mvit-checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "docs" / "results" / "shoplifting-yolo26-v2"
)

SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 113, "val": 18, "test": 30}
FIXED_THRESHOLD = 0.50
BOOTSTRAP_SEED = 20260727
WILSON_Z_95 = 1.959963984540054
SCRIPT_VERSION = "1.1.0"
RULE_FORMULA = (
    "final_risk_i = max(r3d18_mil_probability_i, "
    "yolo26_mvit_mil_probability_i)"
)
RULE_REASON = (
    "CCTV recall-first OR ensemble: alert when either temporal "
    "risk component crosses the fixed 0.50 threshold"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


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
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    temporary.replace(path)


def prediction_index(
    payload: dict[str, Any],
    source_name: str,
) -> dict[str, dict[str, Any]]:
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise RuntimeError(f"{source_name} predictions JSON has no predictions list")
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"{source_name} contains a non-object prediction")
        split = str(row.get("split", ""))
        if split not in SPLITS:
            continue
        relative_path = str(row.get("relative_path", "")).replace("\\", "/")
        if not relative_path:
            raise RuntimeError(f"{source_name} prediction has no relative_path")
        key = relative_path.casefold()
        if key in index:
            raise RuntimeError(f"Duplicate {source_name} prediction: {relative_path}")
        probability = float(row["shoplifting_probability"])
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(
                f"Invalid {source_name} probability for {relative_path}: "
                f"{probability}"
            )
        index[key] = row
    return index


def compare_identity(
    r3d_row: dict[str, Any],
    yolo_row: dict[str, Any],
) -> None:
    keys = ("relative_path", "split", "actor_group", "ground_truth")
    for key in keys:
        left = str(r3d_row.get(key))
        right = str(yolo_row.get(key))
        if key == "relative_path":
            left = left.replace("\\", "/").casefold()
            right = right.replace("\\", "/").casefold()
        if left != right:
            raise RuntimeError(
                f"Component identity mismatch for {key}: {left!r} != {right!r}"
            )


def combine_predictions(
    r3d_payload: dict[str, Any],
    yolo_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    r3d_index = prediction_index(r3d_payload, "R3D-18 MIL")
    yolo_index = prediction_index(yolo_payload, "YOLO-MViT MIL")
    missing_r3d = sorted(set(yolo_index) - set(r3d_index))
    missing_yolo = sorted(set(r3d_index) - set(yolo_index))
    if missing_r3d or missing_yolo:
        raise RuntimeError(
            "Component prediction coverage mismatch: "
            f"missing_r3d={missing_r3d}, missing_yolo={missing_yolo}"
        )
    combined: list[dict[str, Any]] = []
    for key, yolo_row in yolo_index.items():
        r3d_row = r3d_index[key]
        compare_identity(r3d_row, yolo_row)
        r3d_probability = float(r3d_row["shoplifting_probability"])
        yolo_probability = float(yolo_row["shoplifting_probability"])
        final_risk = max(r3d_probability, yolo_probability)
        prediction = int(final_risk >= FIXED_THRESHOLD)
        ground_truth = int(yolo_row["ground_truth"])
        dominant_component = (
            "tie"
            if math.isclose(
                r3d_probability, yolo_probability, rel_tol=0.0, abs_tol=1e-12
            )
            else (
                "r3d18_mil"
                if r3d_probability > yolo_probability
                else "yolo26_mvit_mil"
            )
        )
        combined.append(
            {
                "relative_path": str(yolo_row["relative_path"]).replace("\\", "/"),
                "split": str(yolo_row["split"]),
                "actor_group": str(yolo_row["actor_group"]),
                "ground_truth": ground_truth,
                "ground_truth_name": str(yolo_row["ground_truth_name"]),
                "r3d18_mil_probability": r3d_probability,
                "yolo26_mvit_mil_probability": yolo_probability,
                "final_risk": final_risk,
                "dominant_component": dominant_component,
                "predicted_label": prediction,
                "predicted_name": baseline.LABEL_NAMES[prediction],
                "correct": prediction == ground_truth,
                "threshold": FIXED_THRESHOLD,
                "r3d18_clip_center_fractions": r3d_row.get(
                    "clip_center_fractions", []
                ),
                "r3d18_clip_shoplifting_probabilities": r3d_row.get(
                    "clip_shoplifting_probabilities", []
                ),
                "r3d18_top_2_clip_indices": r3d_row.get(
                    "top_2_clip_indices", []
                ),
                "yolo26_mvit_clip_start_frames": yolo_row.get(
                    "clip_start_frames", []
                ),
                "yolo26_mvit_clip_shoplifting_probabilities": yolo_row.get(
                    "clip_shoplifting_probabilities", []
                ),
                "yolo26_mvit_crop_boxes_xyxy": yolo_row.get(
                    "crop_boxes_xyxy", []
                ),
                "yolo26_mvit_person_found": yolo_row.get("person_found", []),
                "yolo26_mvit_accessory_union": yolo_row.get(
                    "accessory_union", []
                ),
                "yolo26_mvit_selected_view": yolo_row.get("selected_view"),
                "yolo26_mvit_selected_fusion": yolo_row.get("selected_fusion"),
                "yolo26_mvit_selected_pooling": yolo_row.get("selected_pooling"),
            }
        )
    split_order = {name: index for index, name in enumerate(SPLITS)}
    combined.sort(
        key=lambda row: (
            split_order[row["split"]],
            baseline.natural_video_key(row["relative_path"]),
        )
    )
    counts = Counter(row["split"] for row in combined)
    if dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Expected aligned split counts {EXPECTED_COUNTS}, found {dict(counts)}"
        )
    return combined


def metrics_for_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([row["ground_truth"] for row in rows], dtype=np.int64)
    probabilities = np.asarray([row["final_risk"] for row in rows], dtype=np.float64)
    return baseline.binary_metrics(labels, probabilities, FIXED_THRESHOLD)


def component_metrics(
    rows: Sequence[dict[str, Any]],
    probability_field: str,
) -> dict[str, Any]:
    labels = np.asarray([row["ground_truth"] for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [row[probability_field] for row in rows], dtype=np.float64
    )
    return baseline.binary_metrics(labels, probabilities, FIXED_THRESHOLD)


def build_metrics(
    predictions: Sequence[dict[str, Any]],
    bootstrap_iterations: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_split: dict[str, dict[str, Any]] = {}
    component_comparison: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        rows = [row for row in predictions if row["split"] == split]
        metrics = metrics_for_rows(rows)
        by_split[split] = metrics
        component_comparison[split] = {
            "r3d18_mil": component_metrics(rows, "r3d18_mil_probability"),
            "yolo26_mvit_mil": component_metrics(
                rows, "yolo26_mvit_mil_probability"
            ),
            "recall_first_or_ensemble": metrics,
        }
    return by_split, component_comparison


def bootstrap_all_test_metrics(
    rows: Sequence[dict[str, Any]],
    iterations: int,
) -> dict[str, list[float] | None]:
    labels = np.asarray([row["ground_truth"] for row in rows], dtype=np.int64)
    probabilities = np.asarray([row["final_risk"] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices_by_class = [np.where(labels == label)[0] for label in (0, 1)]
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(iterations):
        sample_index = np.concatenate(
            [
                rng.choice(index, size=len(index), replace=True)
                for index in indices_by_class
            ]
        )
        sample_metrics = baseline.binary_metrics(
            labels[sample_index],
            probabilities[sample_index],
            FIXED_THRESHOLD,
        )
        for name in metric_names:
            value = sample_metrics[name]
            if value is not None:
                samples[name].append(float(value))
    return {
        name: (
            [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
            if values
            else None
        )
        for name, values in samples.items()
    }


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    z_squared = WILSON_Z_95**2
    denominator = 1.0 + z_squared / total
    centre = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total**2)
        )
        / denominator
    )
    return [
        max(0.0, float(centre - half_width)),
        min(1.0, float(centre + half_width)),
    ]


def test_wilson_intervals(test_metrics: dict[str, Any]) -> dict[str, list[float] | None]:
    confusion = test_metrics["confusion_matrix"]
    tn = int(confusion["tn"])
    fp = int(confusion["fp"])
    fn = int(confusion["fn"])
    tp = int(confusion["tp"])
    return {
        "accuracy": wilson_interval(tn + tp, tn + fp + fn + tp),
        "precision": wilson_interval(tp, tp + fp),
        "recall": wilson_interval(tp, tp + fn),
        "specificity": wilson_interval(tn, tn + fp),
    }


def flatten_metrics_csv(
    metrics: dict[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
    )
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        item = metrics[split]
        row: dict[str, Any] = {
            "split": split,
            "support": item["support"],
            "normal_support": item["normal_support"],
            "shoplifting_support": item["shoplifting_support"],
            "threshold": FIXED_THRESHOLD,
        }
        for name in metric_names:
            row[name] = item[name]
            interval = item.get("bootstrap_95_ci", {}).get(name)
            row[f"{name}_ci_low"] = interval[0] if interval else ""
            row[f"{name}_ci_high"] = interval[1] if interval else ""
            wilson = item.get("wilson_95_ci", {}).get(name)
            row[f"{name}_wilson_ci_low"] = wilson[0] if wilson else ""
            row[f"{name}_wilson_ci_high"] = wilson[1] if wilson else ""
        rows.append(row)
    fields = (
        "split",
        "support",
        "normal_support",
        "shoplifting_support",
        "threshold",
        *metric_names,
        *(f"{name}_ci_low" for name in metric_names),
        *(f"{name}_ci_high" for name in metric_names),
        *(f"{name}_wilson_ci_low" for name in metric_names),
        *(f"{name}_wilson_ci_high" for name in metric_names),
    )
    return rows, fields


def write_prediction_outputs(
    output_dir: Path,
    predictions: Sequence[dict[str, Any]],
    input_hashes: dict[str, str],
) -> None:
    baseline.write_json(
        output_dir / "predictions.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": utc_now(),
            "selection_rule": RULE_FORMULA,
            "selection_reason": RULE_REASON,
            "fixed_threshold": FIXED_THRESHOLD,
            "test_status": "opened development benchmark",
            "input_sha256": input_hashes,
            "predictions": list(predictions),
        },
    )
    csv_rows: list[dict[str, Any]] = []
    sequence_fields = (
        "r3d18_clip_center_fractions",
        "r3d18_clip_shoplifting_probabilities",
        "r3d18_top_2_clip_indices",
        "yolo26_mvit_clip_start_frames",
        "yolo26_mvit_clip_shoplifting_probabilities",
        "yolo26_mvit_person_found",
        "yolo26_mvit_accessory_union",
    )
    for prediction in predictions:
        row = prediction.copy()
        for field in sequence_fields:
            row[field] = "|".join(str(value) for value in prediction[field])
        row["yolo26_mvit_crop_boxes_xyxy"] = json.dumps(
            prediction["yolo26_mvit_crop_boxes_xyxy"],
            separators=(",", ":"),
        )
        csv_rows.append(row)
    fields = (
        "relative_path",
        "split",
        "actor_group",
        "ground_truth",
        "ground_truth_name",
        "r3d18_mil_probability",
        "yolo26_mvit_mil_probability",
        "final_risk",
        "dominant_component",
        "predicted_label",
        "predicted_name",
        "correct",
        "threshold",
        "r3d18_clip_center_fractions",
        "r3d18_clip_shoplifting_probabilities",
        "r3d18_top_2_clip_indices",
        "yolo26_mvit_clip_start_frames",
        "yolo26_mvit_clip_shoplifting_probabilities",
        "yolo26_mvit_crop_boxes_xyxy",
        "yolo26_mvit_person_found",
        "yolo26_mvit_accessory_union",
        "yolo26_mvit_selected_view",
        "yolo26_mvit_selected_fusion",
        "yolo26_mvit_selected_pooling",
    )
    write_csv(output_dir / "predictions.csv", csv_rows, fields)


def write_confusion_outputs(
    output_dir: Path,
    metrics: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        matrix = metrics[split]["confusion_matrix"]["values"]
        for label, name in baseline.LABEL_NAMES.items():
            rows.append(
                {
                    "split": split,
                    "actual_label": name,
                    "predicted_normal": matrix[label][0],
                    "predicted_shoplifting": matrix[label][1],
                }
            )
    write_csv(
        output_dir / "confusion-matrix.csv",
        rows,
        (
            "split",
            "actual_label",
            "predicted_normal",
            "predicted_shoplifting",
        ),
    )


def render_figures(
    output_dir: Path,
    metrics: dict[str, Any],
    component_comparison: dict[str, dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split_labels = ("Train", "Validation", "Test")
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    figure.patch.set_facecolor("#F8FAFC")
    figure.suptitle(
        "Recall-first OR ensemble confusion matrices",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
    )
    max_value = max(
        int(np.asarray(metrics[split]["confusion_matrix"]["values"]).max())
        for split in SPLITS
    )
    for axis, split, label in zip(axes, SPLITS, split_labels):
        matrix = np.asarray(
            metrics[split]["confusion_matrix"]["values"], dtype=np.int64
        )
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max_value)
        for row_index in range(2):
            for column_index in range(2):
                value = int(matrix[row_index, column_index])
                axis.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color="white" if value > max_value * 0.55 else "#0F172A",
                )
        axis.set_title(label, fontweight="bold")
        axis.set_xticks((0, 1), ("Normal", "Shoplifting"), rotation=22)
        axis.set_yticks((0, 1), ("Normal", "Shoplifting"))
        axis.set_xlabel("Predicted")
        if axis is axes[0]:
            axis.set_ylabel("Ground truth")
    figure.text(
        0.5,
        -0.02,
        "Fixed threshold 0.50 | Test is an opened development benchmark",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    figure.savefig(
        output_dir / "confusion-matrix.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    figure.patch.set_facecolor("#F8FAFC")
    figure.suptitle(
        "Shoplifting recall-first OR ensemble - development benchmark",
        fontsize=19,
        fontweight="bold",
        color="#0F172A",
    )
    colors = {
        "r3d18_mil": "#64748B",
        "yolo26_mvit_mil": "#0EA5E9",
        "recall_first_or_ensemble": "#DC2626",
    }
    names = {
        "r3d18_mil": "R3D-18 MIL",
        "yolo26_mvit_mil": "YOLO26 + MViT",
        "recall_first_or_ensemble": "OR ensemble",
    }

    axis = axes[0, 0]
    x = np.arange(len(SPLITS))
    width = 0.25
    for component_index, component in enumerate(colors):
        values = [
            component_comparison[split][component]["accuracy"]
            for split in SPLITS
        ]
        bars = axis.bar(
            x + (component_index - 1) * width,
            values,
            width,
            color=colors[component],
            label=names[component],
        )
        if component == "recall_first_or_ensemble":
            axis.bar_label(
                bars,
                labels=[f"{value * 100:.1f}%" for value in values],
                fontsize=9,
                padding=3,
            )
    axis.set_xticks(x, split_labels)
    axis.set_ylim(0, 1.20)
    axis.set_ylabel("Accuracy")
    axis.set_title("Component and ensemble accuracy")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        ncol=3,
    )

    axis = axes[0, 1]
    metric_names = ("accuracy", "balanced_accuracy", "recall", "specificity")
    test_values = [metrics["test"][name] for name in metric_names]
    bars = axis.bar(
        np.arange(len(metric_names)),
        test_values,
        color=("#2563EB", "#7C3AED", "#DC2626", "#0891B2"),
    )
    axis.bar_label(
        bars,
        labels=[f"{value * 100:.1f}%" for value in test_values],
        padding=4,
        fontsize=10,
    )
    axis.set_xticks(
        np.arange(len(metric_names)),
        ("Accuracy", "Balanced\naccuracy", "Recall", "Specificity"),
    )
    axis.set_ylim(0, 1.1)
    axis.set_title("Opened test metrics (n=30)")
    axis.grid(axis="y", alpha=0.22)

    axis = axes[1, 0]
    test_rows = [row for row in predictions if row["split"] == "test"]
    for label, marker, color in (
        (0, "o", "#64748B"),
        (1, "^", "#DC2626"),
    ):
        selected = [row for row in test_rows if row["ground_truth"] == label]
        risks = np.asarray([row["final_risk"] for row in selected])
        jitter = np.linspace(-0.12, 0.12, len(selected))
        axis.scatter(
            np.full(len(selected), label) + jitter,
            risks,
            marker=marker,
            s=65,
            alpha=0.85,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=baseline.LABEL_NAMES[label],
        )
    axis.axhline(
        FIXED_THRESHOLD,
        color="#111827",
        linestyle="--",
        linewidth=1.4,
        label="Threshold 0.50",
    )
    axis.set_xticks((0, 1), ("Normal", "Shoplifting"))
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(-0.02, 1.03)
    axis.set_ylabel("Final risk")
    axis.set_title("Opened test risk distribution")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3, fontsize=9)

    axis = axes[1, 1]
    axis.axis("off")
    test = metrics["test"]
    interval = test["wilson_95_ci"]
    text = (
        "Decision rule\n"
        "  final risk = max(R3D-18 MIL, YOLO26 + MViT MIL)\n"
        "  alert when final risk >= 0.50\n\n"
        "Opened test result\n"
        f"  Accuracy             {test['accuracy'] * 100:5.1f}%\n"
        f"  Balanced accuracy    {test['balanced_accuracy'] * 100:5.1f}%\n"
        f"  Recall               {test['recall'] * 100:5.1f}%\n"
        f"  Specificity          {test['specificity'] * 100:5.1f}%\n"
        f"  Confusion             TN {test['tn'] if 'tn' in test else test['confusion_matrix']['tn']}  "
        f"FP {test['confusion_matrix']['fp']}  "
        f"FN {test['confusion_matrix']['fn']}  "
        f"TP {test['confusion_matrix']['tp']}\n"
        f"  Wilson accuracy CI   {interval['accuracy'][0] * 100:.1f}%-"
        f"{interval['accuracy'][1] * 100:.1f}%\n\n"
        "Status\n"
        "  Development benchmark; test opened this round.\n"
        "  A new sealed store/camera/day holdout is required."
    )
    axis.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10.5,
        linespacing=1.35,
        color="#0F172A",
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "white",
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


def checkpoint_manifest(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    yolo_source = resolve_path(args.yolo_checkpoint)
    r3d_checkpoint = resolve_path(args.r3d_checkpoint)
    if not yolo_source.is_file():
        raise FileNotFoundError(yolo_source)
    if not r3d_checkpoint.is_file():
        raise FileNotFoundError(r3d_checkpoint)
    destination = output_dir / "yolo-mvit-checkpoint.pt"
    source_hash = sha256_file(yolo_source)
    if not destination.is_file() or sha256_file(destination) != source_hash:
        shutil.copy2(yolo_source, destination)
    copied_hash = sha256_file(destination)
    if copied_hash != source_hash:
        raise RuntimeError("Copied YOLO-MViT checkpoint hash mismatch")
    payload = torch.load(yolo_source, map_location="cpu", weights_only=True)
    seed_summaries = [
        {
            "seed": item.get("seed"),
            "best_epoch": item.get("best_epoch"),
            "best_validation_score": item.get("best_validation_score"),
        }
        for item in payload.get("seed_checkpoints", [])
    ]
    r3d_reference = Path("..") / "shoplifting" / "checkpoint.pt"
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "ensemble_rule": RULE_FORMULA,
        "yolo_mvit": {
            "file": destination.name,
            "sha256": copied_hash,
            "bytes": destination.stat().st_size,
            "source": str(yolo_source),
            "selected_view": payload.get("selected_view"),
            "selected_fusion": payload.get("selected_fusion"),
            "selected_pooling": payload.get("selected_pooling"),
            "clips_per_video": payload.get("clips_per_video"),
            "seeds": seed_summaries,
            "yolo_model_sha256": payload.get("yolo_model_sha256"),
        },
        "r3d18_mil": {
            "file": r3d_reference.as_posix(),
            "sha256": sha256_file(r3d_checkpoint),
            "bytes": r3d_checkpoint.stat().st_size,
            "storage": "referenced existing checkpoint; not duplicated",
        },
    }
    baseline.write_json(output_dir / "checkpoint.json", manifest)
    return manifest


def benchmark_markdown(
    metrics: dict[str, Any],
    component_comparison: dict[str, dict[str, Any]],
    input_hashes: dict[str, str],
    checkpoint_info: dict[str, Any],
) -> str:
    test = metrics["test"]
    bootstrap_ci = test["bootstrap_95_ci"]
    wilson_ci = test["wilson_95_ci"]
    lines = [
        "# Shoplifting Recall-First OR Ensemble",
        "",
        "Author: Edcosys",
        "",
        "Run date: 27 July 2026",
        "",
        "## Result",
        "",
        (
            "The ensemble takes the elementwise maximum of the R3D-18 MIL and "
            "YOLO26-guided MViT MIL video probabilities. This implements a "
            "CCTV recall-first OR policy at a fixed 0.50 threshold."
        ),
        "",
        "| Split | Videos | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, display_name in (
        ("train", "Train"),
        ("val", "Validation"),
        ("test", "Opened internal test"),
    ):
        item = metrics[split]
        lines.append(
            f"| {display_name} | {item['support']} | "
            f"{item['accuracy'] * 100:.1f}% | "
            f"{item['balanced_accuracy'] * 100:.1f}% | "
            f"{item['precision'] * 100:.1f}% | "
            f"{item['recall'] * 100:.1f}% | "
            f"{item['specificity'] * 100:.1f}% | "
            f"{item['f1'] * 100:.1f}% | "
            f"{item['roc_auc'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Opened-test confusion matrix:",
            "",
            "| Actual class | Predicted Normal | Predicted Shoplifting |",
            "|---|---:|---:|",
            (
                f"| Normal | {test['confusion_matrix']['tn']} | "
                f"{test['confusion_matrix']['fp']} |"
            ),
            (
                f"| Shoplifting | {test['confusion_matrix']['fn']} | "
                f"{test['confusion_matrix']['tp']} |"
            ),
            "",
            (
                f"Test accuracy is **{test['accuracy'] * 100:.1f}%** and recall "
                f"is **{test['recall'] * 100:.1f}%**. Video-level Wilson 95% "
                f"intervals are: accuracy "
                f"{wilson_ci['accuracy'][0] * 100:.1f}%-"
                f"{wilson_ci['accuracy'][1] * 100:.1f}%, recall "
                f"{wilson_ci['recall'][0] * 100:.1f}%-"
                f"{wilson_ci['recall'][1] * 100:.1f}%, specificity "
                f"{wilson_ci['specificity'][0] * 100:.1f}%-"
                f"{wilson_ci['specificity'][1] * 100:.1f}%, and precision "
                f"{wilson_ci['precision'][0] * 100:.1f}%-"
                f"{wilson_ci['precision'][1] * 100:.1f}%."
            ),
            "",
            (
                "The class-stratified 4,000-resample bootstrap 95% intervals "
                "are: accuracy "
                f"{bootstrap_ci['accuracy'][0] * 100:.1f}%-"
                f"{bootstrap_ci['accuracy'][1] * 100:.1f}% and balanced "
                "accuracy "
                f"{bootstrap_ci['balanced_accuracy'][0] * 100:.1f}%-"
                f"{bootstrap_ci['balanced_accuracy'][1] * 100:.1f}%."
            ),
            "",
            "![Metrics dashboard](metrics-dashboard.png)",
            "",
            "![Confusion matrices](confusion-matrix.png)",
            "",
            "## Component comparison",
            "",
            "| Model | Test accuracy | Test balanced accuracy | Test recall | Test specificity |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, display_name in (
        ("r3d18_mil", "R3D-18 MIL"),
        ("yolo26_mvit_mil", "YOLO26-guided MViT MIL"),
        ("recall_first_or_ensemble", "Recall-first OR ensemble"),
    ):
        item = component_comparison["test"][key]
        lines.append(
            f"| {display_name} | {item['accuracy'] * 100:.1f}% | "
            f"{item['balanced_accuracy'] * 100:.1f}% | "
            f"{item['recall'] * 100:.1f}% | "
            f"{item['specificity'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Fixed decision rule",
            "",
            "```text",
            RULE_FORMULA,
            "prediction_i = 1 when final_risk_i >= 0.50",
            "```",
            "",
            "## Network architecture",
            "",
            "```mermaid",
            "flowchart LR",
            '    V["Video"] --> Y["YOLO26"]',
            '    Y --> T["Person / bag tube"]',
            '    T --> C["Cropped-view MViT"]',
            '    V --> F["Full-frame MViT"]',
            '    C --> G["Gated fusion"]',
            "    F --> G",
            '    G --> M["YOLO-MViT risk"]',
            '    V --> R["R3D-18 MIL risk"]',
            '    M --> X["Elementwise max"]',
            "    R --> X",
            '    X --> A["Alert / review"]',
            "```",
            "",
            (
                "The maximum operation is an OR ensemble: either component can "
                "raise an alert. This policy prioritizes theft-event recall for "
                "CCTV triage and accepts the corresponding false-positive risk."
            ),
            "",
            "Each prediction retains both component probabilities, both sets of "
            "clip scores, YOLO crop metadata, the dominant component, and the "
            "final risk.",
            "",
            "## Evaluation status",
            "",
            (
                "**The internal test split was opened during this development "
                "round. The 93.3% result is a development benchmark, not a "
                "sealed final generalization estimate.**"
            ),
            "",
            (
                "Future validation requires a new sealed store-native holdout "
                "grouped by store, camera, day, and incident. It must include "
                "hard negatives and report false alerts per camera-hour."
            ),
            "",
            (
                "The test set contains 30 videos and correlated recording "
                "sessions. The class-stratified bootstrap and video-level "
                "Wilson intervals do not account for session correlation."
            ),
            "",
            "## Reproduction and provenance",
            "",
            "```powershell",
            (
                r".\.venv-yolo\Scripts\python.exe "
                r"scripts\shoplifting_recall_ensemble.py"
            ),
            "```",
            "",
            "Input SHA-256 values:",
            "",
        ]
    )
    for name, digest in input_hashes.items():
        lines.append(f"- `{name}`: `{digest}`")
    lines.extend(
        [
            "",
            "Checkpoint packaging:",
            "",
            (
                f"- YOLO-MViT: `yolo-mvit-checkpoint.pt`, "
                f"{checkpoint_info['yolo_mvit']['bytes']:,} bytes, SHA-256 "
                f"`{checkpoint_info['yolo_mvit']['sha256']}`."
            ),
            (
                "- R3D-18 MIL: "
                f"`{checkpoint_info['r3d18_mil']['file']}`, referenced without "
                "duplication."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the fixed recall-first shoplifting OR ensemble package."
    )
    parser.add_argument(
        "--r3d-predictions", type=Path, default=DEFAULT_R3D_PREDICTIONS
    )
    parser.add_argument("--r3d-metrics", type=Path, default=DEFAULT_R3D_METRICS)
    parser.add_argument(
        "--r3d-checkpoint", type=Path, default=DEFAULT_R3D_CHECKPOINT
    )
    parser.add_argument(
        "--yolo-predictions", type=Path, default=DEFAULT_YOLO_PREDICTIONS
    )
    parser.add_argument("--yolo-summary", type=Path, default=DEFAULT_YOLO_SUMMARY)
    parser.add_argument(
        "--yolo-checkpoint", type=Path, default=DEFAULT_YOLO_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    return parser


def validate_input_metrics(
    predictions: Sequence[dict[str, Any]],
    input_metrics: dict[str, Any],
    probability_field: str,
    source_name: str,
) -> None:
    for split in SPLITS:
        rows = [row for row in predictions if row["split"] == split]
        recomputed = component_metrics(rows, probability_field)
        expected = input_metrics.get("splits", input_metrics.get("metrics", {})).get(
            split
        )
        if expected is None:
            raise RuntimeError(f"{source_name} metrics has no {split} split")
        for metric_name in ("accuracy", "balanced_accuracy", "roc_auc"):
            left = recomputed[metric_name]
            right = expected[metric_name]
            if left is None or right is None:
                if left != right:
                    raise RuntimeError(
                        f"{source_name} {split} {metric_name} mismatch"
                    )
            elif not math.isclose(
                float(left), float(right), rel_tol=0.0, abs_tol=1e-9
            ):
                raise RuntimeError(
                    f"{source_name} {split} {metric_name} mismatch: "
                    f"{left} != {right}"
                )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")
    paths = {
        "r3d_predictions": resolve_path(args.r3d_predictions),
        "r3d_metrics": resolve_path(args.r3d_metrics),
        "yolo_mvit_predictions": resolve_path(args.yolo_predictions),
        "yolo_mvit_summary": resolve_path(args.yolo_summary),
        "yolo_mvit_checkpoint": resolve_path(args.yolo_checkpoint),
        "r3d_checkpoint": resolve_path(args.r3d_checkpoint),
    }
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    r3d_predictions_payload = read_json(paths["r3d_predictions"])
    yolo_predictions_payload = read_json(paths["yolo_mvit_predictions"])
    r3d_metrics_payload = read_json(paths["r3d_metrics"])
    yolo_summary_payload = read_json(paths["yolo_mvit_summary"])
    if not bool(yolo_summary_payload.get("test_accessed")):
        raise RuntimeError("YOLO-MViT input has not explicitly opened the test split")
    predictions = combine_predictions(
        r3d_predictions_payload, yolo_predictions_payload
    )
    validate_input_metrics(
        predictions,
        r3d_metrics_payload,
        "r3d18_mil_probability",
        "R3D-18 MIL",
    )
    validate_input_metrics(
        predictions,
        yolo_summary_payload,
        "yolo26_mvit_mil_probability",
        "YOLO-MViT MIL",
    )
    metrics, component_comparison = build_metrics(
        predictions, args.bootstrap_iterations
    )
    test_rows = [row for row in predictions if row["split"] == "test"]
    metrics["test"]["bootstrap_95_ci"] = bootstrap_all_test_metrics(
        test_rows, args.bootstrap_iterations
    )
    metrics["test"]["bootstrap_iterations"] = args.bootstrap_iterations
    metrics["test"]["bootstrap_method"] = "class-stratified percentile interval"
    metrics["test"]["wilson_95_ci"] = test_wilson_intervals(metrics["test"])
    metrics["test"]["wilson_method"] = (
        "two-sided 95% Wilson score interval at source-video level"
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_info = checkpoint_manifest(args, output_dir)
    config = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "author": "Edcosys",
        "objective": "CCTV recall-first shoplifting risk ensemble",
        "rule_formula": RULE_FORMULA,
        "rule_reason": RULE_REASON,
        "aggregation": "elementwise maximum of aligned source-video probabilities",
        "fixed_threshold": FIXED_THRESHOLD,
        "threshold_tuned": False,
        "component_weight_tuned": False,
        "included_splits": list(SPLITS),
        "test_accessed_this_round": True,
        "test_status": "opened development benchmark",
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_method": "class-stratified percentile interval",
        "wilson_method": "two-sided 95% Wilson score interval at source-video level",
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_sha256": input_hashes,
        "component_configuration": {
            "r3d18_mil": {
                "probability_field": "shoplifting_probability",
                "checkpoint": checkpoint_info["r3d18_mil"]["file"],
            },
            "yolo26_mvit_mil": {
                "probability_field": "shoplifting_probability",
                "selected_view": yolo_summary_payload.get("selected_view"),
                "selected_fusion": yolo_summary_payload.get("selected_fusion"),
                "selected_pooling": yolo_summary_payload.get("selected_pooling"),
                "yolo_sha256": yolo_summary_payload.get("yolo_sha256"),
                "checkpoint": "yolo-mvit-checkpoint.pt",
            },
        },
        "preserved_prediction_fields": [
            "both component probabilities",
            "both component clip scores",
            "R3D clip fractions and top-2 indices",
            "YOLO-MViT clip starts, crop boxes, person detections, and bag unions",
            "dominant component and final risk",
        ],
    }
    baseline.write_json(output_dir / "config.json", config)
    baseline.write_json(
        output_dir / "metrics.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": utc_now(),
            "author": "Edcosys",
            "selection_rule": RULE_FORMULA,
            "fixed_threshold": FIXED_THRESHOLD,
            "test_status": "opened development benchmark",
            "splits": metrics,
            "component_comparison": component_comparison,
        },
    )
    metric_rows, metric_fields = flatten_metrics_csv(metrics)
    write_csv(output_dir / "metrics.csv", metric_rows, metric_fields)
    write_prediction_outputs(output_dir, predictions, input_hashes)
    write_confusion_outputs(output_dir, metrics)
    render_figures(output_dir, metrics, component_comparison, predictions)
    (output_dir / "BENCHMARK.md").write_text(
        benchmark_markdown(
            metrics, component_comparison, input_hashes, checkpoint_info
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rule": RULE_FORMULA,
                "test": metrics["test"],
                "yolo_mvit_checkpoint": checkpoint_info["yolo_mvit"],
                "r3d_checkpoint": checkpoint_info["r3d18_mil"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
