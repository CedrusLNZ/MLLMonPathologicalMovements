#!/usr/bin/env python3
"""
Aggregate segment-level CV predictions into a patient-by-feature CSV.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_PATIENT_REGEX = r"^[^@]+@[^@]+@([^@]+@[^@]+)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate segment_predictions.csv files into a patient-by-feature CSV."
        )
    )
    parser.add_argument(
        "--predictions-root",
        default="finetuned_models",
        help="Root directory containing fold prediction outputs.",
    )
    parser.add_argument(
        "--feature",
        default=None,
        help="Optional feature subfolder to limit aggregation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (defaults to aggregated_patient_predictions.csv).",
    )
    parser.add_argument(
        "--score-column",
        default="prob_1",
        help="Column to aggregate from segment_predictions.csv.",
    )
    parser.add_argument(
        "--agg",
        choices=("max", "mean", "median", "min"),
        default="max",
        help="Aggregation method for patient scores.",
    )
    parser.add_argument(
        "--patient-id-regex",
        default=DEFAULT_PATIENT_REGEX,
        help=(
            "Regex with a capture group for ID in file names "
            "(default keeps sz_v1/sz_v2 separate)."
        ),
    )
    parser.add_argument(
        "--patient-id-source",
        choices=("video_name", "segment_file", "segment_path"),
        default="video_name",
        help="Column to extract patient ID from.",
    )
    parser.add_argument(
        "--patient-csv",
        default="evaluation/dataset/90_FeatureAnnotation.csv",
        help="CSV with patient file names to define row ordering.",
    )
    parser.add_argument(
        "--patient-csv-column",
        default="file_name",
        help="Column in patient CSV that contains file names.",
    )
    parser.add_argument(
        "--missing-value",
        default="",
        help="Value to write when a patient is missing a feature prediction.",
    )
    return parser.parse_args()


def find_prediction_files(root: Path) -> List[Path]:
    return sorted(root.rglob("segment_predictions.csv"))


def infer_feature_and_fold(path: Path) -> Tuple[str, str]:
    fold_dir: Optional[Path] = None
    for parent in path.parents:
        if parent.name.startswith("fold_"):
            fold_dir = parent
            break
    if fold_dir is None:
        return "unknown", "unknown"
    feature_name = fold_dir.parent.name if fold_dir.parent else "unknown"
    return feature_name or "unknown", fold_dir.name


def extract_patient_id(name: str, regex: str) -> str:
    match = re.match(regex, name)
    if match:
        return match.group(1)
    return ""


def aggregate_scores(scores: List[float], method: str) -> float:
    if method == "mean":
        return sum(scores) / len(scores)
    if method == "median":
        return statistics.median(scores)
    if method == "min":
        return min(scores)
    return max(scores)


def iter_rows(paths: Iterable[Path]) -> Iterable[Tuple[Path, Dict[str, str]]]:
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield path, row


def load_patient_ids(
    csv_path: Path, column: str, regex: str
) -> List[str]:
    if not csv_path.exists():
        raise SystemExit(f"patient CSV not found: {csv_path}")
    patient_ids: List[str] = []
    seen: Set[str] = set()
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise SystemExit(
                f"patient CSV missing column '{column}' in {csv_path}"
            )
        for row in reader:
            file_name = row.get(column, "")
            patient_id = extract_patient_id(file_name, regex)
            if not patient_id or patient_id in seen:
                continue
            seen.add(patient_id)
            patient_ids.append(patient_id)
    return sorted(patient_ids)


def main() -> None:
    args = parse_args()

    root = Path(args.predictions_root)
    if args.feature:
        root = root / args.feature

    if not root.exists():
        raise SystemExit(f"predictions root not found: {root}")

    prediction_files = find_prediction_files(root)
    if not prediction_files:
        raise SystemExit(f"no segment_predictions.csv files found under {root}")

    patient_ids = load_patient_ids(
        Path(args.patient_csv), args.patient_csv_column, args.patient_id_regex
    )
    patient_set = set(patient_ids)

    scores_by_patient: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    features: Set[str] = set()
    missing_patients = 0
    unknown_patients = 0
    total_rows = 0

    for path, row in iter_rows(prediction_files):
        total_rows += 1
        if args.score_column not in row:
            raise SystemExit(
                f"missing score column '{args.score_column}' in {path}"
            )
        if args.patient_id_source not in row:
            raise SystemExit(
                f"missing patient ID source '{args.patient_id_source}' in {path}"
            )

        feature_name, _ = infer_feature_and_fold(path)
        if args.feature and feature_name != args.feature:
            continue
        features.add(feature_name)

        source_value = row.get(args.patient_id_source, "")
        patient_id = extract_patient_id(source_value, args.patient_id_regex)
        if not patient_id:
            missing_patients += 1
            continue
        if patient_id not in patient_set:
            unknown_patients += 1
            continue

        try:
            score = float(row[args.score_column])
        except (TypeError, ValueError):
            continue

        key = (feature_name, patient_id)
        scores_by_patient[key].append(score)

    output_path = (
        Path(args.output)
        if args.output
        else Path("aggregated_patient_predictions.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as handle:
        feature_list = sorted(features)
        writer = csv.DictWriter(
            handle,
            fieldnames=["patient_id"] + feature_list,
        )
        writer.writeheader()
        for patient_id in patient_ids:
            row: Dict[str, str | float] = {"patient_id": patient_id}
            for feature_name in feature_list:
                scores = scores_by_patient.get((feature_name, patient_id))
                if scores:
                    row[feature_name] = aggregate_scores(scores, args.agg)
                else:
                    row[feature_name] = args.missing_value
            writer.writerow(row)

    print(
        f"wrote {len(patient_ids)} patients to {output_path} "
        f"(rows={total_rows}, missing_patient_ids={missing_patients}, "
        f"unknown_patient_ids={unknown_patients}, features={len(features)})"
    )


if __name__ == "__main__":
    main()
