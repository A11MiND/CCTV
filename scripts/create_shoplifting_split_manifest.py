"""Create the reviewed actor/clothing-disjoint split for the public dataset.

The public dataset has only clip-level class folders.  It has no actor, session,
camera, or source-take identifiers.  This manifest was created from a visual
audit of all 182 clips, exact SHA-256 duplicate checks, resolution checks, and
grouping by the primary actor's clothing plus visibly connected recording
sessions.

Run from the repository root:

    .venv-yolo\\Scripts\\python.exe scripts\\create_shoplifting_split_manifest.py

The output is consumed by ``scripts/shoplifting_baseline.py --split-manifest``.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_ROOT = WORKSPACE / "data/shoplifting-video-dataset"
OUTPUT_DIR = WORKSPACE / "docs/results/shoplifting"
CSV_PATH = OUTPUT_DIR / "actor-clothing-disjoint-split.csv"
SUMMARY_PATH = OUTPUT_DIR / "split-summary.json"


def ids(*items: int | range) -> set[int]:
    values: set[int] = set()
    for item in items:
        values.update(item if isinstance(item, range) else [item])
    return values


GROUPS: list[dict[str, object]] = [
    {
        "group": "A_checkered",
        "split": "train",
        "normal": ids(1, range(18, 21), 24, range(44, 51), 53),
        "shoplifting": ids(8, 9, 21, 29, range(52, 60)),
        "reason": "Primary checkered-shirt actor and connected takes",
    },
    {
        "group": "B_light_blue",
        "split": "train",
        "normal": ids(range(2, 6), 21, 22, 54, 55, range(57, 65)),
        "shoplifting": ids(range(10, 14), range(60, 71)),
        "reason": "Primary light-blue-shirt actor and connected takes",
    },
    {
        "group": "C_purple",
        "split": "train",
        "normal": ids(6, 17, range(65, 70)),
        "shoplifting": ids(14, 15, 22, range(71, 79)),
        "reason": "Primary purple-shirt actor and connected takes",
    },
    {
        "group": "D_navy_stripe",
        "split": "train",
        "normal": ids(range(7, 11), range(70, 85)),
        "shoplifting": ids(1, range(16, 21), 23, 25, range(79, 88)),
        "reason": "Primary navy/striped-shirt actor and connected takes",
    },
    {
        "group": "E_older_plaid",
        "split": "train",
        "normal": ids(11, 85, 86),
        "shoplifting": set(),
        "reason": "Older plaid-shirt actor; normal-only group",
    },
    {
        "group": "G_brown_plaid",
        "split": "val",
        "normal": ids(30, 37, 38, 88),
        "shoplifting": ids(3, 32, 39, 40),
        "reason": "Primary brown-plaid/masked actor",
    },
    {
        "group": "H_navy_cap",
        "split": "val",
        "normal": ids(23, 35, 36, 43),
        "shoplifting": ids(2, 4, 30, 31, 37, 38),
        "reason": "Primary navy-cap actor and connected takes",
    },
    {
        "group": "F_gray_backpack",
        "split": "test",
        "normal": ids(14, range(31, 35)),
        "shoplifting": ids(range(33, 37)),
        "reason": "Primary gray-plaid/backpack actor",
    },
    {
        "group": "IJK_connected_session",
        "split": "test",
        "normal": ids(15, 16, range(40, 43)),
        "shoplifting": ids(range(5, 8), 24, range(26, 29), range(41, 49), 51),
        "reason": "Red, green, and blue actors co-occur in one connected session",
    },
    {
        "group": "HR_new_scene",
        "split": "ood",
        "normal": ids(89, 90),
        "shoplifting": ids(range(88, 94)),
        "reason": "Locked 1920x1080 new-scene holdout opened after model selection",
    },
    {
        "group": "qualitative_new_angle",
        "split": "qualitative",
        "normal": set(),
        "shoplifting": ids(50),
        "reason": "Positive-only different computer-lab angle; qualitative probe",
    },
    {
        "group": "excluded_context",
        "split": "excluded",
        "normal": ids(12, 13, range(25, 30), 39),
        "shoplifting": set(),
        "reason": "Context/background-only or unsuitable active-person evidence",
    },
    {
        "group": "excluded_exact_duplicate",
        "split": "excluded",
        "normal": ids(51, 52, 56),
        "shoplifting": set(),
        "reason": "Bit-identical duplicate; retained sources are N50, N44, and N55",
    },
    {
        "group": "excluded_cross_group",
        "split": "excluded",
        "normal": ids(87),
        "shoplifting": set(),
        "reason": "Multiple primary actors cross the reviewed train/test groups",
    },
]


def numeric_id(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[1])


def main() -> None:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)
    assignments: dict[str, dict[str, str]] = {}
    for group in GROUPS:
        for class_name in ("normal", "shoplifting"):
            for identifier in group[class_name]:
                relative_path = f"{class_name}/{class_name}-{identifier}.mp4"
                if relative_path in assignments:
                    raise RuntimeError(f"Duplicate assignment: {relative_path}")
                assignments[relative_path] = {
                    "relative_path": relative_path,
                    "split": str(group["split"]),
                    "actor_group": str(group["group"]),
                    "reason": str(group["reason"]),
                }

    discovered = {
        path.relative_to(DATA_ROOT).as_posix()
        for path in DATA_ROOT.rglob("*.mp4")
    }
    missing = sorted(discovered - set(assignments))
    extra = sorted(set(assignments) - discovered)
    if missing or extra:
        raise RuntimeError(f"Manifest mismatch: missing={missing}, extra={extra}")
    if len(assignments) != 182:
        raise RuntimeError(f"Expected 182 assignments, found {len(assignments)}")

    rows = sorted(
        assignments.values(),
        key=lambda row: (
            row["relative_path"].split("/", 1)[0],
            numeric_id(Path(row["relative_path"])),
        ),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("relative_path", "split", "actor_group", "reason"),
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((row["split"], row["relative_path"].split("/", 1)[0]) for row in rows)
    summary = {
        "schema_version": 1,
        "dataset": "MNNIT Allahabad Shoplifting Dataset, DOI 10.17632/r3yjf35hzr.1",
        "review_unit": "source video",
        "total_videos": len(rows),
        "split_counts": {
            split: {
                "normal": counts[(split, "normal")],
                "shoplifting": counts[(split, "shoplifting")],
                "total": counts[(split, "normal")] + counts[(split, "shoplifting")],
            }
            for split in ("train", "val", "test", "ood", "qualitative", "excluded")
        },
        "method": (
            "Manual primary-actor/clothing and connected-session grouping, exact "
            "SHA-256 duplicate quarantine, and a locked high-resolution scene holdout"
        ),
        "limitations": [
            "The source dataset does not provide actor or session identifiers.",
            "The low-resolution core retains the same laboratory and recurring bystanders.",
            "The OOD split has only two Normal and six Shoplifting clips.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(CSV_PATH)
    print(json.dumps(summary["split_counts"], indent=2))


if __name__ == "__main__":
    main()
