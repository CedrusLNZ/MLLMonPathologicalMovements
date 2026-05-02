#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.io import read_video, read_video_timestamps
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVideoClassification


DEFAULT_MODEL = "google/vivit-b-16x2-kinetics400"
DEFAULT_EXTENSIONS = ".mp4,.avi,.mov,.mkv"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_RESAMPLE_FPS = 2.0
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_PATIENT_REGEX = r"^([A-Z]\d+)@"
DEFAULT_SEGMENT_REGEX = r"^(.*)_segment_\d+$"


def load_annotations(csv_path: str, name_column: str) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    duplicates = 0
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        if name_column not in reader.fieldnames:
            raise ValueError(
                f"CSV missing required file-name column '{name_column}'. "
                f"Available columns: {', '.join(reader.fieldnames)}"
            )
        for row in reader:
            filename = (row.get(name_column) or "").strip()
            if not filename:
                continue
            if filename in mapping:
                duplicates += 1
                continue
            mapping[filename] = row
    if duplicates:
        print(f"warning: skipped {duplicates} duplicate entries in CSV.", file=sys.stderr)
    return mapping, reader.fieldnames


def list_video_files(root: str, extensions: Tuple[str, ...]) -> List[str]:
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                paths.append(os.path.join(dirpath, filename))
    return sorted(paths)


def parse_label(value: str) -> int:
    text = (value or "").strip().lower()
    return 1 if text == "yes" else 0


def derive_base_video_name(file_name: str, segment_pattern: re.Pattern[str]) -> str:
    stem, ext = os.path.splitext(file_name)
    match = segment_pattern.match(stem)
    if match and match.lastindex:
        return f"{match.group(1)}{ext}"
    return file_name


def build_segment_index(
    segment_paths: List[str],
    segment_pattern: re.Pattern[str],
) -> Dict[str, List[str]]:
    segments_by_video: Dict[str, List[str]] = {}
    for path in segment_paths:
        base_name = derive_base_video_name(os.path.basename(path), segment_pattern)
        segments_by_video.setdefault(base_name, []).append(path)
    for paths in segments_by_video.values():
        paths.sort()
    return segments_by_video


def extract_patient_id(file_name: str, regex: str) -> str:
    match = re.match(regex, file_name)
    if match:
        return match.group(1)
    return ""


def compute_resample_indices(
    total_frames: int,
    original_fps: float,
    target_fps: float,
) -> torch.Tensor:
    if total_frames <= 0:
        return torch.zeros(0, dtype=torch.long)
    if not target_fps or target_fps <= 0:
        return torch.arange(total_frames)
    if not original_fps or original_fps <= 0 or original_fps <= target_fps:
        return torch.arange(total_frames)
    step = original_fps / target_fps
    indices = torch.arange(0, total_frames, step)
    indices = torch.round(indices).long()
    indices = torch.clamp(indices, 0, total_frames - 1)
    indices = torch.unique_consecutive(indices)
    if indices.numel() == 0:
        indices = torch.tensor([0], dtype=torch.long)
    return indices


def estimate_resampled_frame_count(path: str, resample_fps: float) -> int:
    total_frames = 0
    fps = None
    try:
        timestamps, fps = read_video_timestamps(path, pts_unit="sec")
        total_frames = len(timestamps)
    except Exception:
        total_frames = 0
    if total_frames == 0:
        try:
            video, _audio, info = read_video(path, pts_unit="sec")
            total_frames = video.shape[0]
            if isinstance(info, dict):
                fps = info.get("video_fps")
        except Exception as exc:
            print(
                f"warning: unable to read video for frame count '{path}': {exc}",
                file=sys.stderr,
            )
            return 0
    indices = compute_resample_indices(total_frames, fps or 0.0, resample_fps)
    return int(indices.numel())


def estimate_max_resampled_frames(paths: List[str], resample_fps: float) -> int:
    max_frames = 0
    for path in paths:
        frame_count = estimate_resampled_frame_count(path, resample_fps)
        if frame_count > max_frames:
            max_frames = frame_count
    return max_frames


class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        num_frames: int,
        resample_fps: float,
        image_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        training: bool,
    ) -> None:
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.resample_fps = resample_fps
        self.image_size = image_size
        self.training = training
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.video_paths[idx]
        label = self.labels[idx]
        video, _audio, info = read_video(path, pts_unit="sec")
        if video.numel() == 0:
            raise RuntimeError(f"Empty video returned for {path}")
        video = video.permute(0, 3, 1, 2)  # T, C, H, W
        video = self._resample_frames(video, info)
        video = self._pad_or_trim_frames(video)
        video = self._resize_and_normalize(video)
        return {
            "pixel_values": video,
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def _resample_frames(self, video: torch.Tensor, info: Dict[str, float]) -> torch.Tensor:
        fps = info.get("video_fps") if isinstance(info, dict) else None
        indices = compute_resample_indices(video.shape[0], fps or 0.0, self.resample_fps)
        if indices.numel() == 0:
            raise RuntimeError("Unable to compute resample indices for video.")
        return video.index_select(0, indices)

    def _pad_or_trim_frames(self, video: torch.Tensor) -> torch.Tensor:
        if self.num_frames <= 0:
            return video
        total_frames = video.shape[0]
        if total_frames >= self.num_frames:
            if self.training:
                start = random.randint(0, total_frames - self.num_frames)
            else:
                start = (total_frames - self.num_frames) // 2
            indices = torch.arange(start, start + self.num_frames)
            return video.index_select(0, indices)
        pad = self.num_frames - total_frames
        last = video[-1:].repeat(pad, 1, 1, 1)
        return torch.cat([video, last], dim=0)

    def _resize_and_normalize(self, video: torch.Tensor) -> torch.Tensor:
        video = video.float() / 255.0
        video = F.interpolate(
            video,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return (video - self.mean) / self.std


def get_classifier_module(model: nn.Module) -> nn.Module:
    for attr in ("classifier", "head", "fc"):
        if hasattr(model, attr):
            module = getattr(model, attr)
            if isinstance(module, nn.Module):
                return module
    raise ValueError(
        "Unable to locate classifier head on model; "
        "use --train-full to finetune the entire model."
    )


def split_dataset(
    paths: List[str],
    labels: List[int],
    val_split: float,
    seed: int,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    if val_split <= 0 or len(paths) < 2:
        return list(paths), list(labels), [], []
    indices = list(range(len(paths)))
    random.Random(seed).shuffle(indices)
    split_idx = int(len(indices) * (1.0 - val_split))
    split_idx = max(1, min(split_idx, len(indices) - 1))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    return train_paths, train_labels, val_paths, val_labels


def build_kfolds(indices: List[int], k: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    folds = [[] for _ in range(k)]
    for idx, item in enumerate(shuffled):
        folds[idx % k].append(item)
    return folds


def build_stratified_folds(
    indices: List[int],
    labels: List[int],
    k: int,
    seed: int,
) -> Tuple[List[List[int]], bool]:
    rng = random.Random(seed)
    label_groups = {0: [], 1: []}
    for idx in indices:
        label_groups[int(labels[idx])].append(idx)
    if min(len(label_groups[0]), len(label_groups[1])) < k:
        return build_kfolds(indices, k, seed), False
    folds = [[] for _ in range(k)]
    for label in (0, 1):
        group = label_groups[label]
        rng.shuffle(group)
        for idx, item in enumerate(group):
            folds[idx % k].append(item)
    for fold in folds:
        rng.shuffle(fold)
    return folds, True


def derive_patient_label(labels: List[int], policy: str) -> int:
    if policy == "any":
        return 1 if any(label == 1 for label in labels) else 0
    if policy == "all":
        return 1 if all(label == 1 for label in labels) else 0
    if policy == "majority":
        return 1 if sum(labels) / max(len(labels), 1) > 0.5 else 0
    raise ValueError(f"Unknown patient label policy: {policy}")


def stratified_patient_split(
    patient_ids: List[str],
    patient_labels: List[int],
    val_split: float,
    seed: int,
) -> Tuple[List[str], List[str], bool]:
    rng = random.Random(seed)
    label_groups = {0: [], 1: []}
    for patient_id, label in zip(patient_ids, patient_labels):
        label_groups[int(label)].append(patient_id)

    if min(len(label_groups[0]), len(label_groups[1])) < 2:
        shuffled = list(patient_ids)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1.0 - val_split))
        split_idx = max(1, min(split_idx, len(shuffled) - 1))
        return shuffled[:split_idx], shuffled[split_idx:], False

    rng.shuffle(label_groups[0])
    rng.shuffle(label_groups[1])

    def val_count(total: int) -> int:
        count = int(round(total * val_split))
        return max(1, min(count, total - 1))

    val_pos = val_count(len(label_groups[1]))
    val_neg = val_count(len(label_groups[0]))
    val_ids = label_groups[1][:val_pos] + label_groups[0][:val_neg]
    train_ids = label_groups[1][val_pos:] + label_groups[0][val_neg:]
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids, True


def split_dataset_by_patient(
    paths: List[str],
    labels: List[int],
    val_split: float,
    seed: int,
    patient_regex: str,
    label_policy: str,
) -> Tuple[List[str], List[int], List[str], List[int], Dict[str, int], bool]:
    patient_map: Dict[str, Dict[str, List[int]]] = {}
    missing_patient = 0
    for idx, (path, label) in enumerate(zip(paths, labels)):
        file_name = os.path.basename(path)
        patient_id = extract_patient_id(file_name, patient_regex)
        if not patient_id:
            missing_patient += 1
            patient_id = f"unknown::{file_name}"
        entry = patient_map.setdefault(patient_id, {"indices": [], "labels": []})
        entry["indices"].append(idx)
        entry["labels"].append(label)

    patient_ids = []
    patient_labels = []
    for patient_id, entry in patient_map.items():
        patient_ids.append(patient_id)
        patient_labels.append(derive_patient_label(entry["labels"], label_policy))

    if val_split <= 0 or len(patient_ids) < 2:
        train_patients = list(patient_ids)
        val_patients = []
        stratified = False
    else:
        train_patients, val_patients, stratified = stratified_patient_split(
            patient_ids, patient_labels, val_split, seed
        )

    train_paths: List[str] = []
    train_labels: List[int] = []
    for patient_id in train_patients:
        for idx in patient_map[patient_id]["indices"]:
            train_paths.append(paths[idx])
            train_labels.append(labels[idx])

    val_paths: List[str] = []
    val_labels: List[int] = []
    for patient_id in val_patients:
        for idx in patient_map[patient_id]["indices"]:
            val_paths.append(paths[idx])
            val_labels.append(labels[idx])

    stats = {
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "missing_patient_ids": missing_patient,
    }
    return train_paths, train_labels, val_paths, val_labels, stats, stratified


def build_patient_folds(
    paths: List[str],
    labels: List[int],
    k: int,
    seed: int,
    patient_regex: str,
    label_policy: str,
) -> Tuple[List[List[str]], Dict[str, Dict[str, List[int]]], Dict[str, int], bool]:
    patient_map: Dict[str, Dict[str, List[int]]] = {}
    missing_patient = 0
    for idx, (path, label) in enumerate(zip(paths, labels)):
        file_name = os.path.basename(path)
        patient_id = extract_patient_id(file_name, patient_regex)
        if not patient_id:
            missing_patient += 1
            patient_id = f"unknown::{file_name}"
        entry = patient_map.setdefault(patient_id, {"indices": [], "labels": []})
        entry["indices"].append(idx)
        entry["labels"].append(label)

    patient_ids = []
    patient_labels = []
    for patient_id, entry in patient_map.items():
        patient_ids.append(patient_id)
        patient_labels.append(derive_patient_label(entry["labels"], label_policy))

    if k > len(patient_ids):
        return [], patient_map, {"missing_patient_ids": missing_patient}, False

    rng = random.Random(seed)
    label_groups = {0: [], 1: []}
    for patient_id, label in zip(patient_ids, patient_labels):
        label_groups[int(label)].append(patient_id)

    stratified = False
    if min(len(label_groups[0]), len(label_groups[1])) >= k:
        stratified = True
        folds = [[] for _ in range(k)]
        for label in (0, 1):
            group = label_groups[label]
            rng.shuffle(group)
            for idx, patient_id in enumerate(group):
                folds[idx % k].append(patient_id)
    else:
        shuffled = list(patient_ids)
        rng.shuffle(shuffled)
        folds = [[] for _ in range(k)]
        for idx, patient_id in enumerate(shuffled):
            folds[idx % k].append(patient_id)

    stats = {
        "total_patients": len(patient_ids),
        "missing_patient_ids": missing_patient,
    }
    return folds, patient_map, stats, stratified


def collect_paths_labels(
    patient_ids: List[str],
    patient_map: Dict[str, Dict[str, List[int]]],
    paths: List[str],
    labels: List[int],
) -> Tuple[List[str], List[int]]:
    split_paths: List[str] = []
    split_labels: List[int] = []
    for patient_id in patient_ids:
        for idx in patient_map[patient_id]["indices"]:
            split_paths.append(paths[idx])
            split_labels.append(labels[idx])
    return split_paths, split_labels


def resolve_image_processor_params(
    model_name: str,
    cache_dir: str | None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], int]:
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception as exc:
        print(
            f"warning: unable to load image processor for '{model_name}': {exc}; "
            "falling back to default normalization.",
            file=sys.stderr,
        )
        return MEAN, STD, DEFAULT_IMAGE_SIZE

    mean = tuple(getattr(processor, "image_mean", MEAN))
    std = tuple(getattr(processor, "image_std", STD))
    size = getattr(processor, "size", DEFAULT_IMAGE_SIZE)
    image_size = DEFAULT_IMAGE_SIZE
    if isinstance(size, dict):
        if "height" in size and "width" in size and size["height"] == size["width"]:
            image_size = size["height"]
        elif "shortest_edge" in size:
            image_size = size["shortest_edge"]
    elif isinstance(size, int):
        image_size = size
    return mean, std, image_size


def load_model(
    model_name: str,
    num_labels: int,
    num_frames: int,
    override_num_frames: bool,
    cache_dir: str | None,
) -> nn.Module:
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    if hasattr(config, "num_frames") and override_num_frames and num_frames > 0:
        config.num_frames = num_frames
    if hasattr(config, "num_labels"):
        config.num_labels = num_labels
    return AutoModelForVideoClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


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
    processor_image_size: int,
    num_frames: int,
    image_size: int,
    extra_summary: Dict[str, object] | None = None,
) -> Dict[str, object]:
    model = load_model(
        args.model_name,
        num_labels=2,
        num_frames=num_frames,
        override_num_frames=args.use_all_frames,
        cache_dir=args.cache_dir,
    )
    if not args.use_all_frames and hasattr(model.config, "num_frames"):
        if model.config.num_frames != num_frames:
            print(
                f"warning: model expects {model.config.num_frames} frames; "
                f"overriding --num-frames to match.",
                file=sys.stderr,
            )
            num_frames = model.config.num_frames

    model_image_size = getattr(model.config, "image_size", None)
    if isinstance(model_image_size, (tuple, list)):
        if len(model_image_size) == 2 and model_image_size[0] == model_image_size[1]:
            model_image_size = model_image_size[0]
    if isinstance(model_image_size, int) and model_image_size != image_size:
        print(
            f"warning: model expects {model_image_size} image size; "
            f"overriding --image-size to match.",
            file=sys.stderr,
        )
        image_size = model_image_size
    elif processor_image_size != image_size:
        print(
            f"warning: image processor expects {processor_image_size} image size; "
            f"overriding --image-size to match.",
            file=sys.stderr,
        )
        image_size = processor_image_size

    if not args.train_full:
        for param in model.parameters():
            param.requires_grad = False
        classifier = get_classifier_module(model)
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
        "train_full": args.train_full,
        "train_size": len(train_paths),
        "val_size": len(val_paths),
        "split_by_patient": args.split_by_patient,
        "patient_label_policy": args.patient_label_policy if args.split_by_patient else None,
        "best_val_acc": best_val_acc,
        "model_name": args.model_name,
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


@torch.no_grad()
def predict_segments_to_csv(
    model: nn.Module,
    segment_paths: List[str],
    output_csv: str,
    segment_pattern: re.Pattern[str],
    args: argparse.Namespace,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    image_size: int,
    num_frames: int,
) -> Dict[str, int]:
    dataset = VideoFolderDataset(
        segment_paths,
        [0 for _ in segment_paths],
        num_frames=num_frames,
        resample_fps=args.resample_fps,
        image_size=image_size,
        mean=mean,
        std=std,
        training=False,
    )
    device = torch.device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "segment_path",
                "segment_file",
                "video_name",
                "prob_0",
                "prob_1",
                "pred_label",
            ],
        )
        writer.writeheader()
        offset = 0
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=-1).cpu()
            batch_paths = segment_paths[offset : offset + probs.shape[0]]
            offset += probs.shape[0]
            for path, prob in zip(batch_paths, probs):
                writer.writerow(
                    {
                        "segment_path": path,
                        "segment_file": os.path.basename(path),
                        "video_name": derive_base_video_name(
                            os.path.basename(path), segment_pattern
                        ),
                        "prob_0": float(prob[0].item()),
                        "prob_1": float(prob[1].item()),
                        "pred_label": int(prob.argmax().item()),
                    }
                )

    return {"segments": len(segment_paths)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Finetune a pretrained video ViT for each subfolder dataset and save models."
        )
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
        "--model-name",
        default=DEFAULT_MODEL,
        help="Pretrained model name or local path.",
    )
    parser.add_argument(
        "--output-dir",
        default="finetuned_models",
        help="Output directory for finetuned models.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Directory for Hugging Face caches (defaults to <output-dir>/hf_cache)."
        ),
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
    if args.cache_dir:
        args.cache_dir = os.path.abspath(args.cache_dir)
    else:
        args.cache_dir = os.path.join(output_root, "hf_cache")
    os.makedirs(args.cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(args.cache_dir, "hub")
    os.environ["HF_ASSETS_CACHE"] = os.path.join(args.cache_dir, "assets")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(args.cache_dir, "datasets")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.cache_dir, "transformers")

    annotations, columns = load_annotations(args.csv, args.csv_name_column)
    mean, std, processor_image_size = resolve_image_processor_params(
        args.model_name, args.cache_dir
    )
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
                        mean,
                        std,
                        processor_image_size,
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
                    model = AutoModelForVideoClassification.from_pretrained(
                        fold_output, cache_dir=args.cache_dir
                    )
                    model.to(device)
                    model.eval()
                    model_num_frames = getattr(
                        model.config, "num_frames", dataset_num_frames
                    )
                    model_image_size = getattr(
                        model.config, "image_size", dataset_image_size
                    )
                    if isinstance(model_image_size, (tuple, list)):
                        if (
                            len(model_image_size) == 2
                            and model_image_size[0] == model_image_size[1]
                        ):
                            model_image_size = model_image_size[0]
                    if not isinstance(model_image_size, int):
                        model_image_size = dataset_image_size
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
                        mean,
                        std,
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
                            mean,
                            std,
                            processor_image_size,
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
                            mean,
                            std,
                            processor_image_size,
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
            mean,
            std,
            processor_image_size,
            dataset_num_frames,
            dataset_image_size,
            extra_summary=extra_summary,
        )
        print(f"saved finetuned model for '{feature_name}' to {output_path}")


if __name__ == "__main__":
    main()
