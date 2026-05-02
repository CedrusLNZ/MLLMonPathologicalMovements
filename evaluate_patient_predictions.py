#!/usr/bin/env python3
"""
Evaluate aggregated patient predictions against the feature annotation CSV.

Outputs two metric files:
- Fixed-threshold metrics (default 0.5).
- Tuned-threshold metrics (per feature, maximize F1 or Youden's J).

Optional CV mode runs k-fold splits on patient IDs, tuning thresholds on
k-1 folds and evaluating on the held-out fold.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_PATIENT_REGEX = r"^[^@]+@[^@]+@([^@]+@[^@]+)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Threshold aggregated patient predictions and compute metrics "
            "against the feature annotation CSV."
        )
    )
    parser.add_argument(
        "--predictions",
        default="aggregated_patient_predictions.csv",
        help="CSV with aggregated patient predictions.",
    )
    parser.add_argument(
        "--ground-truth",
        default="evaluation/dataset/90_FeatureAnnotation.csv",
        help="CSV with ground-truth labels (yes/no).",
    )
    parser.add_argument(
        "--ground-truth-file-column",
        default="file_name",
        help="Column in ground-truth CSV that contains file names.",
    )
    parser.add_argument(
        "--patient-id-regex",
        default=DEFAULT_PATIENT_REGEX,
        help="Regex with a capture group for patient ID in file names.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.5,
        help="Fixed threshold for logits/probabilities.",
    )
    parser.add_argument(
        "--tuning-patients-csv",
        default=None,
        help=(
            "Optional CSV containing patient IDs for threshold tuning "
            "(fairer calibration)."
        ),
    )
    parser.add_argument(
        "--tuning-patient-id-column",
        default="patient_id",
        help="Column name in tuning patients CSV.",
    )
    parser.add_argument(
        "--tuning-strategy",
        choices=("f1", "youden"),
        default="f1",
        help="Metric to maximize when selecting tuned thresholds.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help=(
            "If >1, run k-fold CV with thresholds tuned on k-1 folds and "
            "evaluated on the held-out fold."
        ),
    )
    parser.add_argument(
        "--cv-seed",
        type=int,
        default=42,
        help="Random seed for CV patient splits.",
    )
    parser.add_argument(
        "--cv-output-dir",
        default="cv_metrics",
        help="Output directory for per-fold metrics when CV is enabled.",
    )
    parser.add_argument(
        "--output-fixed",
        default="metrics_threshold_0p5.csv",
        help="Output CSV for fixed-threshold metrics.",
    )
    parser.add_argument(
        "--output-tuned",
        default="metrics_threshold_tuned.csv",
        help="Output CSV for tuned-threshold metrics.",
    )
    return parser.parse_args()


def extract_patient_id(name: str, regex: str) -> str:
    match = re.match(regex, name)
    if match:
        return match.group(1)
    return ""


def normalize_label(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true", "t"}:
        return 1
    if text in {"no", "n", "0", "false", "f"}:
        return 0
    return None


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_confusion(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Tuple[int, int, int, int]:
    preds = scores >= threshold
    labels = labels.astype(bool)
    tp = int(np.sum(preds & labels))
    fp = int(np.sum(preds & ~labels))
    tn = int(np.sum(~preds & ~labels))
    fn = int(np.sum(~preds & labels))
    return tp, fp, tn, fn


def compute_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Dict[str, float]:
    tp, fp, tn, fn = compute_confusion(scores, labels, threshold)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "support": int(tp + tn + fp + fn),
        "positives": int(tp + fn),
        "negatives": int(tn + fp),
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(scores)
    if unique_scores.size == 0:
        return np.array([0.5])
    if unique_scores.size == 1:
        return np.array([unique_scores[0]])
    # Add slight padding to allow all-negative/all-positive decisions.
    padding = 1e-6
    return np.concatenate(
        (
            np.array([unique_scores[0] - padding]),
            unique_scores,
            np.array([unique_scores[-1] + padding]),
        )
    )


def pick_best_threshold(
    scores: np.ndarray, labels: np.ndarray, strategy: str
) -> float:
    thresholds = candidate_thresholds(scores)
    best_threshold = thresholds[0]
    best_score = -1.0
    for threshold in thresholds:
        tp, fp, tn, fn = compute_confusion(scores, labels, threshold)
        if strategy == "youden":
            recall = safe_div(tp, tp + fn)
            specificity = safe_div(tn, tn + fp)
            score = recall + specificity - 1.0
        else:
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            score = safe_div(2 * precision * recall, precision + recall)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def load_tuning_patient_ids(
    csv_path: Optional[str], column: str
) -> Optional[set]:
    if not csv_path:
        return None
    path = Path(csv_path)
    if not path.exists():
        raise SystemExit(f"tuning patients CSV not found: {path}")
    df = pd.read_csv(path)
    if column not in df.columns:
        raise SystemExit(
            f"tuning patients CSV missing column '{column}' in {path}"
        )
    return set(df[column].dropna().astype(str).tolist())


def aggregate_ground_truth(
    gt_path: Path,
    file_column: str,
    patient_regex: str,
    feature_cols: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    gt_raw = pd.read_csv(gt_path)
    if file_column not in gt_raw.columns:
        raise SystemExit(
            f"ground-truth CSV missing column '{file_column}' in {gt_path}"
        )
    gt_raw["patient_id"] = gt_raw[file_column].apply(
        lambda value: extract_patient_id(str(value), patient_regex)
    )
    gt_raw = gt_raw[gt_raw["patient_id"] != ""]

    available_features = [f for f in feature_cols if f in gt_raw.columns]
    conflict_counts: Dict[str, int] = {}
    for feature in available_features:
        if feature not in gt_raw.columns:
            continue
        gt_raw[feature] = gt_raw[feature].apply(normalize_label)

        def has_conflict(values: pd.Series) -> bool:
            unique_values = set(v for v in values.dropna().tolist())
            return len(unique_values) > 1

        conflicts = (
            gt_raw.groupby("patient_id")[feature]
            .apply(has_conflict)
            .sum()
        )
        conflict_counts[feature] = int(conflicts)

    def collapse(values: pd.Series) -> Optional[int]:
        unique_values = list(set(v for v in values.dropna().tolist()))
        if len(unique_values) == 1:
            return int(unique_values[0])
        return None

    aggregated = gt_raw.groupby("patient_id")[available_features].agg(collapse)
    return aggregated, conflict_counts


def evaluate_feature(
    feature: str,
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    fixed_threshold: float,
    tuning_ids: Optional[set],
    eval_ids: Optional[set],
    tuning_strategy: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    data = pd.concat(
        [pred_df[feature], gt_df[feature]],
        axis=1,
        keys=["score", "label"],
    ).dropna()

    if data.empty:
        empty_metrics = {
            "feature": feature,
            "threshold": float("nan"),
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "f1": 0.0,
            "support": 0,
            "positives": 0,
            "negatives": 0,
        }
        return empty_metrics, empty_metrics

    if tuning_ids is None:
        tuning_data = data
    else:
        tuning_data = data.loc[data.index.isin(tuning_ids)]

    if eval_ids is None:
        if tuning_ids is None:
            eval_data = data
        else:
            eval_data = data.loc[~data.index.isin(tuning_ids)]
            if eval_data.empty:
                eval_data = data
    else:
        eval_data = data.loc[data.index.isin(eval_ids)]

    eval_scores = eval_data["score"].to_numpy(dtype=float)
    eval_labels = eval_data["label"].to_numpy(dtype=int)

    fixed_metrics = compute_metrics(eval_scores, eval_labels, fixed_threshold)
    fixed_metrics.update({"feature": feature, "threshold": fixed_threshold})

    if tuning_data.empty:
        tuned_threshold = fixed_threshold
    else:
        tuning_scores = tuning_data["score"].to_numpy(dtype=float)
        tuning_labels = tuning_data["label"].to_numpy(dtype=int)
        tuned_threshold = pick_best_threshold(
            tuning_scores, tuning_labels, tuning_strategy
        )

    tuned_metrics = compute_metrics(
        eval_scores, eval_labels, tuned_threshold
    )
    tuned_metrics.update(
        {
            "feature": feature,
            "threshold": tuned_threshold,
        }
    )
    return fixed_metrics, tuned_metrics


def build_cv_folds(
    patient_ids: List[str], n_folds: int, seed: int
) -> List[List[str]]:
    ids = np.array(sorted(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return [list(chunk) for chunk in np.array_split(ids, n_folds)]


def summarize_cv_metrics(
    frames: List[pd.DataFrame],
    fixed_threshold: Optional[float],
) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    metric_cols = ["precision", "recall", "accuracy", "f1"]
    count_cols = ["support", "positives", "negatives"]

    summary = (
        combined.groupby("feature", as_index=False)[metric_cols]
        .mean()
        .merge(
            combined.groupby("feature", as_index=False)[count_cols].sum(),
            on="feature",
        )
        .merge(
            combined.groupby("feature", as_index=False)
            .size()
            .rename(columns={"size": "folds"}),
            on="feature",
        )
    )

    if fixed_threshold is not None:
        summary["threshold"] = fixed_threshold
    else:
        threshold_mean = (
            combined.groupby("feature", as_index=False)["threshold"]
            .mean()
        )
        summary = summary.merge(threshold_mean, on="feature")
    return summary


def main() -> None:
    args = parse_args()

    pred_path = Path(args.predictions)
    gt_path = Path(args.ground_truth)
    if not pred_path.exists():
        raise SystemExit(f"predictions CSV not found: {pred_path}")
    if not gt_path.exists():
        raise SystemExit(f"ground-truth CSV not found: {gt_path}")

    pred_df = pd.read_csv(pred_path)
    if "patient_id" not in pred_df.columns:
        raise SystemExit(
            f"predictions CSV missing 'patient_id' in {pred_path}"
        )
    pred_df = pred_df.set_index("patient_id")
    pred_df = pred_df.apply(pd.to_numeric, errors="coerce")

    pred_features = [c for c in pred_df.columns if c != "patient_id"]
    gt_df, conflict_counts = aggregate_ground_truth(
        gt_path,
        args.ground_truth_file_column,
        args.patient_id_regex,
        pred_features,
    )

    common_features = [f for f in pred_features if f in gt_df.columns]
    if not common_features:
        raise SystemExit("no overlapping features between predictions and ground truth")

    patient_ids = sorted(set(pred_df.index) & set(gt_df.index))

    if args.cv_folds and args.cv_folds > 1:
        if args.tuning_patients_csv:
            print("note: --tuning-patients-csv ignored because CV is enabled")
        if len(patient_ids) < args.cv_folds:
            raise SystemExit("not enough patients for requested CV folds")

        output_dir = Path(args.cv_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        folds = build_cv_folds(patient_ids, args.cv_folds, args.cv_seed)

        fixed_fold_frames: List[pd.DataFrame] = []
        tuned_fold_frames: List[pd.DataFrame] = []

        for fold_idx, eval_ids in enumerate(folds):
            eval_set = set(eval_ids)
            tuning_set = set(patient_ids) - eval_set
            fixed_rows: List[Dict[str, float]] = []
            tuned_rows: List[Dict[str, float]] = []

            for feature in common_features:
                fixed_metrics, tuned_metrics = evaluate_feature(
                    feature,
                    pred_df,
                    gt_df,
                    args.fixed_threshold,
                    tuning_set,
                    eval_set,
                    args.tuning_strategy,
                )
                fixed_metrics["fold"] = fold_idx
                tuned_metrics["fold"] = fold_idx
                tuned_metrics["tuning_strategy"] = args.tuning_strategy
                fixed_rows.append(fixed_metrics)
                tuned_rows.append(tuned_metrics)

            fold_fixed = pd.DataFrame(fixed_rows).sort_values("feature")
            fold_tuned = pd.DataFrame(tuned_rows).sort_values("feature")

            fold_fixed_path = output_dir / (
                f"metrics_threshold_fixed_fold_{fold_idx}.csv"
            )
            fold_tuned_path = output_dir / (
                f"metrics_threshold_tuned_fold_{fold_idx}.csv"
            )
            fold_fixed.to_csv(fold_fixed_path, index=False)
            fold_tuned.to_csv(fold_tuned_path, index=False)

            fixed_fold_frames.append(fold_fixed)
            tuned_fold_frames.append(fold_tuned)

            print(
                "wrote fold {fold} metrics to {fixed} and {tuned}".format(
                    fold=fold_idx,
                    fixed=fold_fixed_path,
                    tuned=fold_tuned_path,
                )
            )

        fixed_mean = summarize_cv_metrics(
            fixed_fold_frames, fixed_threshold=args.fixed_threshold
        )
        tuned_mean = summarize_cv_metrics(tuned_fold_frames, fixed_threshold=None)
        tuned_mean["tuning_strategy"] = args.tuning_strategy

        fixed_mean.to_csv(args.output_fixed, index=False)
        tuned_mean.to_csv(args.output_tuned, index=False)

        print(f"wrote CV mean fixed metrics to {args.output_fixed}")
        print(f"wrote CV mean tuned metrics to {args.output_tuned}")
    else:
        tuning_ids = load_tuning_patient_ids(
            args.tuning_patients_csv, args.tuning_patient_id_column
        )

        fixed_rows = []
        tuned_rows = []
        for feature in common_features:
            fixed_metrics, tuned_metrics = evaluate_feature(
                feature,
                pred_df,
                gt_df,
                args.fixed_threshold,
                tuning_ids,
                None,
                args.tuning_strategy,
            )
            fixed_rows.append(fixed_metrics)
            tuned_metrics["tuning_strategy"] = args.tuning_strategy
            tuned_rows.append(tuned_metrics)

        fixed_out = pd.DataFrame(fixed_rows).sort_values("feature")
        tuned_out = pd.DataFrame(tuned_rows).sort_values("feature")

        fixed_out.to_csv(args.output_fixed, index=False)
        tuned_out.to_csv(args.output_tuned, index=False)

        print(f"wrote fixed metrics to {args.output_fixed}")
        print(f"wrote tuned metrics to {args.output_tuned}")
        if tuning_ids is None:
            print(
                "note: tuned thresholds were selected on the full dataset "
                "(provide --tuning-patients-csv for a held-out calibration set)."
            )

    conflict_total = sum(count for count in conflict_counts.values())
    if conflict_total:
        print(
            "note: conflicting labels detected for some patients/features; "
            "those entries were dropped."
        )


if __name__ == "__main__":
    main()
