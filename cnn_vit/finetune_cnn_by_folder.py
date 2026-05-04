#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import random
import re
import statistics
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import video as video_models

from finetune_vit_by_folder import (
    DEFAULT_EXTENSIONS,
    DEFAULT_PATIENT_REGEX,
    DEFAULT_RESAMPLE_FPS,
    DEFAULT_SEGMENT_REGEX,
    MEAN,
    STD,
    VideoFolderDataset,
    build_kfolds,
    build_patient_folds,
    build_segment_index,
    build_stratified_folds,
    collect_paths_labels,
    derive_base_video_name,
    estimate_max_resampled_frames,
    evaluate,
    get_classifier_module,
    list_video_files,
    load_annotations,
    parse_label,
    predict_segments_to_csv,
    split_dataset,
    split_dataset_by_patient,
    train_one_epoch,
)


DEFAULT_CNN_ARCH = "r3d_18"
DEFAULT_IMAGE_SIZE = 112
CNN_CONFIG_NAME = "cnn_config.json"
CNN_WEIGHTS_NAME = "model.pt"

SUPPORTED_CNN_ARCHS = ("r3d_18", "mc3_18", "r2plus1d_18")
WEIGHTS_BY_ARCH = {
    "r3d_18": "R3D_18_Weights",
    "mc3_18": "MC3_18_Weights",
    "r2plus1d_18": "R2Plus1D_18_Weights",
}


class CnnOutput:
    def __init__(self, loss: torch.Tensor | None, logits: torch.Tensor) -> None:
        self.loss = loss
        self.logits = logits


class CnnVideoClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, config: SimpleNamespace) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config

    def forward(  # type: ignore[override]
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> CnnOutput:
        if pixel_values.ndim != 5:
            raise ValueError(
                f"Expected pixel_values to be 5D (B,T,C,H,W); got {pixel_values.shape}"
            )
        # Convert (B, T, C, H, W) -> (B, C, T, H, W) for 3D CNNs.
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        logits = self.backbone(pixel_values)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return CnnOutput(loss, logits)

    def save_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, CNN_WEIGHTS_NAME))
        with open(os.path.join(output_dir, CNN_CONFIG_NAME), "w") as handle:
            json.dump(
                {
                    "arch": self.config.arch,
                    "num_labels": self.config.num_labels,
                    "num_frames": self.config.num_frames,
                    "image_size": self.config.image_size,
                },
                handle,
                indent=2,
            )


def build_cnn_backbone(arch: str, pretrained: bool) -> nn.Module:
    if not hasattr(video_models, arch):
        raise ValueError(
            f"Unknown CNN architecture '{arch}'. "
            f"Available: {', '.join(SUPPORTED_CNN_ARCHS)}"
        )
    model_fn = getattr(video_models, arch)
    signature = None
    try:
        signature = inspect.signature(model_fn)
    except (TypeError, ValueError):
        signature = None

    if signature and "weights" in signature.parameters:
        weights = None
        if pretrained:
            weights_name = WEIGHTS_BY_ARCH.get(arch)
            weights_enum = (
                getattr(video_models, weights_name, None)
                if weights_name
                else None
            )
            if weights_enum is None:
                print(
                    f"warning: pretrained weights not available for '{arch}'.",
                    file=sys.stderr,
                )
            else:
                weights = weights_enum.DEFAULT
        return model_fn(weights=weights)

    return model_fn(pretrained=pretrained)


def replace_classifier(backbone: nn.Module, num_labels: int) -> None:
    for attr in ("classifier", "head", "fc"):
        if not hasattr(backbone, attr):
            continue
        module = getattr(backbone, attr)
        if isinstance(module, nn.Linear):
            setattr(backbone, attr, nn.Linear(module.in_features, num_labels))
            return
        if isinstance(module, nn.Sequential) and module:
            last = module[-1]
            if isinstance(last, nn.Linear):
                module[-1] = nn.Linear(last.in_features, num_labels)
                return
    raise ValueError("Unable to replace classifier head on CNN model.")


def build_cnn_model(
    arch: str,
    num_labels: int,
    pretrained: bool,
    num_frames: int,
    image_size: int,
) -> CnnVideoClassifier:
    backbone = build_cnn_backbone(arch, pretrained)
    replace_classifier(backbone, num_labels)
    config = SimpleNamespace(
        arch=arch,
        num_labels=num_labels,
        num_frames=num_frames,
        image_size=image_size,
    )
    return CnnVideoClassifier(backbone, config)


def load_cnn_from_dir(model_dir: str, device: torch.device) -> CnnVideoClassifier:
    config_path = os.path.join(model_dir, CNN_CONFIG_NAME)
    weights_path = os.path.join(model_dir, CNN_WEIGHTS_NAME)
    if not os.path.exists(config_path):
        raise ValueError(f"Missing CNN config: {config_path}")
    if not os.path.exists(weights_path):
        raise ValueError(f"Missing CNN weights: {weights_path}")
    with open(config_path) as handle:
        config = json.load(handle)
    model = build_cnn_model(
        arch=config.get("arch", DEFAULT_CNN_ARCH),
        num_labels=int(config.get("num_labels", 2)),
        pretrained=False,
        num_frames=int(config.get("num_frames", 0)),
        image_size=int(config.get("image_size", DEFAULT_IMAGE_SIZE)),
    )
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    return model


def train_feature_split(
    feature_name: str,
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    output_path: str,
    args: argparse.Namespace,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    num_frames: int,
    image_size: int,
    extra_summary: Dict[str, object] | None = None,
) -> Dict[str, object]:
    model = build_cnn_model(
        args.arch,
        num_labels=2,
        pretrained=args.pretrained,
        num_frames=num_frames,
        image_size=image_size,
    )

    if not args.train_full:
        for param in model.parameters():
            param.requires_grad = False
        classifier = get_classifier_module(model.backbone)
        for param in classifier.parameters():
            param.requires_grad = True

    device = torch.device(args.device)
    model.to(device)

    train_dataset = VideoFolderDataset(
        train_paths,
        train_labels,
        num_frames=num_frames,
        resample_fps=args.resample_fps,
        image_size=image_size,
        mean=mean,
        std=std,
        training=True,
    )
    val_dataset = (
        VideoFolderDataset(
            val_paths,
            val_labels,
            num_frames=num_frames,
            resample_fps=args.resample_fps,
            image_size=image_size,
            mean=mean,
            std=std,
            training=False,
        )
        if val_paths
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_dataset
        else None
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(
                f"[{feature_name}] epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            print(
                f"[{feature_name}] epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
            )

    os.makedirs(output_path, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.save_pretrained(output_path)

    summary = {
        "feature_name": feature_name,
        "arch": args.arch,
        "pretrained": args.pretrained,
        "train_full": args.train_full,
        "train_size": len(train_paths),
        "val_size": len(val_paths),
        "split_by_patient": args.split_by_patient,
        "patient_label_policy": args.patient_label_policy if args.split_by_patient else None,
        "best_val_acc": best_val_acc,
        "num_frames": num_frames,
        "image_size": image_size,
        "resample_fps": args.resample_fps,
        "use_all_frames": args.use_all_frames,
    }
    if extra_summary:
        summary.update(extra_summary)
    with open(os.path.join(output_path, "training_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finetune a pretrained CNN for each subfolder dataset."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing one subfolder per feature dataset.",
    )
    parser.add_argument(
        "--csv",
        default="evaluation/dataset/90_FeatureAnnotation.csv",
        help="Path to the annotation CSV.",
    )
    parser.add_argument(
        "--csv-name-column",
        default="file_name",
        help="CSV column name that contains the video filename.",
    )
    parser.add_argument(
        "--arch",
        default=DEFAULT_CNN_ARCH,
        choices=SUPPORTED_CNN_ARCHS,
        help="CNN video architecture to finetune.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained Kinetics weights when available.",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Disable pretrained weights.",
    )
    parser.add_argument(
        "--output-dir",
        default="finetuned_models_cnn",
        help="Output directory for finetuned models.",
    )
    parser.add_argument(
        "--train-full",
        action="store_true",
        help="Finetune the full model (default: only classifier head).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs per dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="Number of frames sampled per video.",
    )
    parser.add_argument(
        "--resample-fps",
        type=float,
        default=DEFAULT_RESAMPLE_FPS,
        help="Target FPS for temporal downsampling (0 to disable).",
    )
    parser.add_argument(
        "--use-all-frames",
        action="store_true",
        default=True,
        help="Use all frames after resampling by setting num-frames to the max per dataset.",
    )
    parser.add_argument(
        "--no-use-all-frames",
        dest="use_all_frames",
        action="store_false",
        help="Disable using all frames after resampling.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Spatial size for frames after resizing.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Number of folds for k-fold CV (0 or 1 for a single train/val split).",
    )
    parser.add_argument(
        "--split-by-patient",
        action="store_true",
        help="Split train/val by patient ID to avoid leakage.",
    )
    parser.add_argument(
        "--patient-id-regex",
        default=DEFAULT_PATIENT_REGEX,
        help="Regex with a capture group for patient ID in file names.",
    )
    parser.add_argument(
        "--patient-label-policy",
        choices=("any", "majority", "all"),
        default="any",
        help="How to derive patient labels for stratification.",
    )
    parser.add_argument(
        "--extensions",
        default=DEFAULT_EXTENSIONS,
        help="Comma-separated list of video extensions to include.",
    )
    parser.add_argument(
        "--segments-root",
        default=None,
        help=(
            "Root directory containing all 30s segments for CV inference "
            "(enables segment-based CV when used with --cv-folds > 1)."
        ),
    )
    parser.add_argument(
        "--segment-regex",
        default=DEFAULT_SEGMENT_REGEX,
        help=(
            "Regex applied to a segment filename stem to recover the base video name. "
            "Must contain a capture group for the base name."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = os.path.abspath(args.data_root)
    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    annotations, columns = load_annotations(args.csv, args.csv_name_column)
    try:
        segment_pattern = re.compile(args.segment_regex)
    except re.error as exc:
        raise ValueError(f"Invalid --segment-regex: {exc}") from exc
    extensions = tuple(
        ext.strip().lower()
        for ext in args.extensions.split(",")
        if ext.strip()
    )
    if not extensions:
        raise ValueError("No extensions provided.")

    dataset_folders = [
        entry
        for entry in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, entry))
    ]
    if not dataset_folders:
        raise ValueError(f"No dataset folders found under {data_root}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    base_num_frames = args.num_frames
    base_image_size = args.image_size
    if args.cv_folds and args.cv_folds > 1 and args.val_split > 0:
        print(
            "warning: --val-split is ignored when --cv-folds > 1.",
            file=sys.stderr,
        )

    run_segment_cv = bool(args.cv_folds and args.cv_folds > 1 and args.segments_root)
    segment_folds: List[List[str]] = []
    segments_by_video: Dict[str, List[str]] = {}
    if run_segment_cv:
        if args.split_by_patient:
            print(
                "warning: --split-by-patient is ignored when using --segments-root.",
                file=sys.stderr,
            )
        segment_root = os.path.abspath(args.segments_root)
        if not os.path.isdir(segment_root):
            raise ValueError(f"--segments-root does not exist: {segment_root}")
        segment_paths = list_video_files(segment_root, extensions)
        if not segment_paths:
            raise ValueError(f"No segment videos found under {segment_root}")
        segments_by_video = build_segment_index(segment_paths, segment_pattern)
        fold_videos = sorted(annotations.keys())
        if len(fold_videos) < args.cv_folds:
            raise ValueError(
                f"Not enough videos for {args.cv_folds}-fold CV "
                f"(found {len(fold_videos)})."
            )
        fold_indices = build_kfolds(
            list(range(len(fold_videos))), args.cv_folds, args.seed
        )
        segment_folds = [[fold_videos[i] for i in fold] for fold in fold_indices]
        folds_dir = os.path.join(output_root, "cv_folds")
        os.makedirs(folds_dir, exist_ok=True)
        with open(os.path.join(folds_dir, "folds.json"), "w") as handle:
            json.dump(
                {
                    "folds": args.cv_folds,
                    "seed": args.seed,
                    "segment_regex": args.segment_regex,
                    "folds_videos": segment_folds,
                },
                handle,
                indent=2,
            )
        for idx, fold in enumerate(segment_folds, start=1):
            with open(os.path.join(folds_dir, f"fold_{idx}.txt"), "w") as handle:
                handle.write("\n".join(fold))
        with open(os.path.join(folds_dir, "all_videos.txt"), "w") as handle:
            handle.write("\n".join(fold_videos))

    for feature_name in dataset_folders:
        if feature_name not in columns:
            print(
                f"warning: folder '{feature_name}' not found in CSV columns; skipping.",
                file=sys.stderr,
            )
            continue

        folder_path = os.path.join(data_root, feature_name)
        video_paths = list_video_files(folder_path, extensions)
        labels: List[int] = []
        matched_paths: List[str] = []
        matched_base_names: List[str] = []
        missing = 0
        for path in video_paths:
            filename = os.path.basename(path)
            base_name = derive_base_video_name(filename, segment_pattern)
            row = annotations.get(base_name)
            if row is None:
                missing += 1
                continue
            label = parse_label(row.get(feature_name, ""))
            matched_paths.append(path)
            labels.append(label)
            matched_base_names.append(base_name)

        if missing:
            print(
                f"warning: {missing} videos in '{feature_name}' missing from CSV.",
                file=sys.stderr,
            )

        if len(matched_paths) < 2:
            print(
                f"warning: not enough labeled videos for '{feature_name}'; skipping.",
                file=sys.stderr,
            )
            continue

        dataset_num_frames = base_num_frames
        if args.use_all_frames:
            max_frames = estimate_max_resampled_frames(
                matched_paths, args.resample_fps
            )
            if max_frames <= 0:
                print(
                    f"warning: unable to estimate max frames for '{feature_name}'; "
                    "falling back to --num-frames.",
                    file=sys.stderr,
                )
            else:
                dataset_num_frames = max_frames
                print(
                    f"[{feature_name}] using all frames after resampling; "
                    f"num_frames={dataset_num_frames}"
                )

        dataset_image_size = base_image_size
        run_cv = args.cv_folds and args.cv_folds > 1
        if run_cv:
            feature_output_root = os.path.join(output_root, feature_name)
            os.makedirs(feature_output_root, exist_ok=True)
            fold_summaries = []
            if run_segment_cv:
                for fold_idx in range(args.cv_folds):
                    val_videos = segment_folds[fold_idx]
                    train_videos = [
                        video_name
                        for idx, fold in enumerate(segment_folds)
                        if idx != fold_idx
                        for video_name in fold
                    ]
                    train_video_set = set(train_videos)
                    train_paths = [
                        path
                        for path, base_name in zip(matched_paths, matched_base_names)
                        if base_name in train_video_set
                    ]
                    train_labels = [
                        label
                        for label, base_name in zip(labels, matched_base_names)
                        if base_name in train_video_set
                    ]
                    if not train_paths:
                        print(
                            f"warning: fold {fold_idx + 1} empty for '{feature_name}'; skipping.",
                            file=sys.stderr,
                        )
                        continue
                    print(
                        f"[{feature_name}] fold {fold_idx + 1}/{args.cv_folds} "
                        f"train_videos={len(train_videos)} "
                        f"train_segments={len(train_paths)} "
                        f"test_videos={len(val_videos)}"
                    )
                    fold_output = os.path.join(
                        feature_output_root, f"fold_{fold_idx + 1}"
                    )
                    extra_summary = {
                        "fold_index": fold_idx + 1,
                        "folds": args.cv_folds,
                        "train_videos": len(train_videos),
                        "test_videos": len(val_videos),
                        "segment_cv": True,
                    }
                    summary = train_feature_split(
                        feature_name,
                        train_paths,
                        train_labels,
                        [],
                        [],
                        fold_output,
                        args,
                        MEAN,
                        STD,
                        dataset_num_frames,
                        dataset_image_size,
                        extra_summary=extra_summary,
                    )
                    fold_summaries.append(summary)
                    print(
                        f"saved finetuned model for '{feature_name}' fold {fold_idx + 1} "
                        f"to {fold_output}"
                    )

                    test_segments = [
                        path
                        for video_name in val_videos
                        for path in segments_by_video.get(video_name, [])
                    ]
                    if not test_segments:
                        print(
                            f"warning: no segments found for fold {fold_idx + 1} "
                            f"test videos in '{feature_name}'.",
                            file=sys.stderr,
                        )
                        continue
                    model = load_cnn_from_dir(fold_output, device)
                    model.eval()
                    model_num_frames = getattr(
                        model.config, "num_frames", dataset_num_frames
                    )
                    model_image_size = getattr(
                        model.config, "image_size", dataset_image_size
                    )
                    predictions_dir = os.path.join(fold_output, "predictions")
                    predictions_path = os.path.join(
                        predictions_dir, "segment_predictions.csv"
                    )
                    pred_stats = predict_segments_to_csv(
                        model,
                        test_segments,
                        predictions_path,
                        segment_pattern,
                        args,
                        MEAN,
                        STD,
                        model_image_size,
                        model_num_frames,
                    )
                    with open(
                        os.path.join(predictions_dir, "prediction_summary.json"), "w"
                    ) as handle:
                        json.dump(
                            {
                                "fold_index": fold_idx + 1,
                                "segments": pred_stats["segments"],
                                "test_videos": len(val_videos),
                                "segment_regex": args.segment_regex,
                            },
                            handle,
                            indent=2,
                        )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                continue
            cv_ready = True
            folds: List[List[int]] | List[List[str]] = []
            stratified = False
            patient_map: Dict[str, Dict[str, List[int]]] = {}
            fold_stats: Dict[str, int] = {}

            if args.split_by_patient:
                folds, patient_map, fold_stats, stratified = build_patient_folds(
                    matched_paths,
                    labels,
                    args.cv_folds,
                    args.seed,
                    args.patient_id_regex,
                    args.patient_label_policy,
                )
                if not folds:
                    print(
                        f"warning: insufficient patients for {args.cv_folds}-fold CV "
                        f"in '{feature_name}'; using --val-split.",
                        file=sys.stderr,
                    )
                    cv_ready = False
                elif fold_stats.get("missing_patient_ids"):
                    print(
                        f"warning: {fold_stats['missing_patient_ids']} videos in "
                        f"'{feature_name}' had no patient ID match; treated as unique patients.",
                        file=sys.stderr,
                    )
            else:
                if len(matched_paths) < args.cv_folds:
                    print(
                        f"warning: insufficient samples for {args.cv_folds}-fold CV "
                        f"in '{feature_name}'; using --val-split.",
                        file=sys.stderr,
                    )
                    cv_ready = False
                else:
                    indices = list(range(len(matched_paths)))
                    folds, stratified = build_stratified_folds(
                        indices, labels, args.cv_folds, args.seed
                    )

            if cv_ready:
                if args.split_by_patient:
                    for fold_idx in range(args.cv_folds):
                        val_patients = folds[fold_idx]
                        train_patients = [
                            patient_id
                            for idx, fold in enumerate(folds)
                            if idx != fold_idx
                            for patient_id in fold
                        ]
                        train_paths, train_labels = collect_paths_labels(
                            train_patients, patient_map, matched_paths, labels
                        )
                        val_paths, val_labels = collect_paths_labels(
                            val_patients, patient_map, matched_paths, labels
                        )
                        if not train_paths or not val_paths:
                            print(
                                f"warning: fold {fold_idx + 1} empty for '{feature_name}'; skipping.",
                                file=sys.stderr,
                            )
                            continue
                        print(
                            f"[{feature_name}] fold {fold_idx + 1}/{args.cv_folds} "
                            f"train_patients={len(train_patients)} "
                            f"val_patients={len(val_patients)} "
                            f"train_videos={len(train_paths)} "
                            f"val_videos={len(val_paths)} "
                            f"stratified={stratified}"
                        )
                        fold_output = os.path.join(
                            feature_output_root, f"fold_{fold_idx + 1}"
                        )
                        extra_summary = {
                            "fold_index": fold_idx + 1,
                            "folds": args.cv_folds,
                            "train_patients": len(train_patients),
                            "val_patients": len(val_patients),
                            "patient_stratified": stratified,
                        }
                        summary = train_feature_split(
                            feature_name,
                            train_paths,
                            train_labels,
                            val_paths,
                            val_labels,
                            fold_output,
                            args,
                            MEAN,
                            STD,
                            dataset_num_frames,
                            dataset_image_size,
                            extra_summary=extra_summary,
                        )
                        fold_summaries.append(summary)
                        print(
                            f"saved finetuned model for '{feature_name}' fold {fold_idx + 1} "
                            f"to {fold_output}"
                        )
                else:
                    for fold_idx in range(args.cv_folds):
                        val_idx = folds[fold_idx]
                        train_idx = [
                            idx
                            for fold_id, fold in enumerate(folds)
                            if fold_id != fold_idx
                            for idx in fold
                        ]
                        train_paths = [matched_paths[i] for i in train_idx]
                        train_labels = [labels[i] for i in train_idx]
                        val_paths = [matched_paths[i] for i in val_idx]
                        val_labels = [labels[i] for i in val_idx]
                        if not train_paths or not val_paths:
                            print(
                                f"warning: fold {fold_idx + 1} empty for '{feature_name}'; skipping.",
                                file=sys.stderr,
                            )
                            continue
                        print(
                            f"[{feature_name}] fold {fold_idx + 1}/{args.cv_folds} "
                            f"train_videos={len(train_paths)} val_videos={len(val_paths)} "
                            f"stratified={stratified}"
                        )
                        fold_output = os.path.join(
                            feature_output_root, f"fold_{fold_idx + 1}"
                        )
                        extra_summary = {
                            "fold_index": fold_idx + 1,
                            "folds": args.cv_folds,
                            "cv_stratified": stratified,
                        }
                        summary = train_feature_split(
                            feature_name,
                            train_paths,
                            train_labels,
                            val_paths,
                            val_labels,
                            fold_output,
                            args,
                            MEAN,
                            STD,
                            dataset_num_frames,
                            dataset_image_size,
                            extra_summary=extra_summary,
                        )
                        fold_summaries.append(summary)
                        print(
                            f"saved finetuned model for '{feature_name}' fold {fold_idx + 1} "
                            f"to {fold_output}"
                        )

                if fold_summaries:
                    val_scores = [summary["best_val_acc"] for summary in fold_summaries]
                    mean_acc = statistics.mean(val_scores)
                    std_acc = (
                        statistics.pstdev(val_scores) if len(val_scores) > 1 else 0.0
                    )
                    cv_summary = {
                        "feature_name": feature_name,
                        "folds": args.cv_folds,
                        "split_by_patient": args.split_by_patient,
                        "patient_label_policy": args.patient_label_policy
                        if args.split_by_patient
                        else None,
                        "mean_best_val_acc": mean_acc,
                        "std_best_val_acc": std_acc,
                        "folds_metrics": val_scores,
                        "num_frames": dataset_num_frames,
                        "image_size": dataset_image_size,
                        "resample_fps": args.resample_fps,
                        "use_all_frames": args.use_all_frames,
                    }
                    with open(
                        os.path.join(feature_output_root, "cv_summary.json"), "w"
                    ) as handle:
                        json.dump(cv_summary, handle, indent=2)
                continue

        split_stats = {}
        stratified = False
        if args.split_by_patient:
            (
                train_paths,
                train_labels,
                val_paths,
                val_labels,
                split_stats,
                stratified,
            ) = split_dataset_by_patient(
                matched_paths,
                labels,
                args.val_split,
                args.seed,
                args.patient_id_regex,
                args.patient_label_policy,
            )
        else:
            train_paths, train_labels, val_paths, val_labels = split_dataset(
                matched_paths, labels, args.val_split, args.seed
            )
        if not train_paths:
            print(
                f"warning: training split empty for '{feature_name}'; skipping.",
                file=sys.stderr,
            )
            continue
        if args.split_by_patient and split_stats.get("missing_patient_ids"):
            print(
                f"warning: {split_stats['missing_patient_ids']} videos in '{feature_name}' "
                "had no patient ID match; treated as unique patients.",
                file=sys.stderr,
            )
        if args.split_by_patient:
            print(
                f"[{feature_name}] train_patients={split_stats.get('train_patients')} "
                f"val_patients={split_stats.get('val_patients')} "
                f"train_videos={len(train_paths)} val_videos={len(val_paths)} "
                f"stratified={stratified}"
            )

        output_path = os.path.join(output_root, feature_name)
        extra_summary = {
            "train_patients": split_stats.get("train_patients"),
            "val_patients": split_stats.get("val_patients"),
            "patient_stratified": stratified if args.split_by_patient else None,
        }
        train_feature_split(
            feature_name,
            train_paths,
            train_labels,
            val_paths,
            val_labels,
            output_path,
            args,
            MEAN,
            STD,
            dataset_num_frames,
            dataset_image_size,
            extra_summary=extra_summary,
        )
        print(f"saved finetuned model for '{feature_name}' to {output_path}")


if __name__ == "__main__":
    main()
