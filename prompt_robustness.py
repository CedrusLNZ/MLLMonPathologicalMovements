#!/usr/bin/env python3
"""Prompt robustness and prompt-optimization experiment utilities.

This script builds fold-aware video classification manifests, runs prompt
variants through an external video-MLLM backend, and scores prompt sensitivity.
It also includes an optional GEPA hook for optimizing prompts with a black-box
backend command.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_FEATURES = (
    "arm_flexion",
    "asynchronous_movement",
    "oral_automatisms",
)

FEATURE_LABELS = {
    "arm_flexion": "arm flexion",
    "arm_straightening": "arm straightening",
    "asynchronous_movement": "asynchronous movement",
    "oral_automatisms": "oral automatisms",
    "blank_stare": "blank stare",
    "close_eyes": "closed eyes",
    "head_turning": "head turning",
    "tonic": "tonic stiffening",
    "clonic": "clonic jerking",
    "full_body_shaking": "full-body shaking",
}

FEATURE_DESCRIPTIONS = {
    "arm_flexion": "bending or flexing an arm at the elbow or shoulder",
    "arm_straightening": (
        "straightening or extending one or both arms at the elbow or shoulder"
    ),
    "asynchronous_movement": (
        "left and right limbs or body parts moving at different times rather "
        "than in a synchronized rhythm"
    ),
    "oral_automatisms": (
        "repetitive mouth or oral movements such as chewing, lip smacking, "
        "swallowing, or mouthing"
    ),
    "blank_stare": (
        "a sustained blank stare or reduced visual engagement during the event"
    ),
    "close_eyes": (
        "the eyes being closed for a sustained part of the event rather than "
        "only brief blinking"
    ),
    "head_turning": (
        "visible turning or version of the head to one side during the event"
    ),
    "tonic": (
        "sustained stiffening, increased muscle tone, or rigid posturing of "
        "the body or limbs"
    ),
    "clonic": (
        "repeated rhythmic jerking movements of the face, trunk, or limbs"
    ),
    "full_body_shaking": (
        "shaking or jerking movements involving most of the body rather than "
        "only one isolated limb or body part"
    ),
}

SEED_PROMPT_TEMPLATES = {
    "minimal": {
        "source": "minimal",
        "text": "Does this video show {feature_label}? Answer yes or no.",
    },
    "brief_definition": {
        "source": "non_expert_detail",
        "text": (
            "Does this video show {feature_label} ({feature_description})? "
            "Answer yes or no."
        ),
    },
    "ilae_informed": {
        "source": "ilae_informed",
        "text_by_feature": {
            "arm_flexion": (
                "Using seizure semiology terminology, look for visible arm "
                "flexion: bending of either arm at the elbow or shoulder as "
                "part of the ictal motor behavior, including tonic, dystonic, "
                "clonic, or asymmetric posturing. Does the video show arm "
                "flexion? Answer yes or no."
            ),
            "asynchronous_movement": (
                "Using seizure semiology terminology, look for asynchronous "
                "or asymmetric motor activity: body parts or left and right "
                "limbs moving out of phase, at different times, or with "
                "different rhythm, frequency, or amplitude rather than as a "
                "synchronized bilateral movement. Does the video show "
                "asynchronous movement? Answer yes or no."
            ),
            "oral_automatisms": (
                "Using the ILAE concept of automatisms as coordinated, "
                "repetitive motor activity that can resemble voluntary "
                "movement, look for oroalimentary/oral automatisms such as "
                "chewing, lip smacking, mouthing, tongue movements, or "
                "swallowing. Does the video show oral automatisms? Answer yes "
                "or no."
            ),
            "arm_straightening": (
                "Using seizure semiology terminology, look for visible arm "
                "extension or straightening: one or both arms becoming "
                "straightened or extended at the elbow or shoulder as part of "
                "the ictal motor behavior, including tonic, dystonic, or "
                "asymmetric posturing. Does the video show arm straightening? "
                "Answer yes or no."
            ),
            "blank_stare": (
                "Using seizure semiology terminology, look for behavioral "
                "arrest or impaired awareness visible as a sustained blank "
                "stare, fixed gaze, or reduced visual engagement during the "
                "event. Does the video show a blank stare? Answer yes or no."
            ),
            "close_eyes": (
                "Using seizure semiology terminology, look for eye closure "
                "during the event: the patient's eyes remain closed for a "
                "sustained period rather than only brief eye blinking. Does "
                "the video show closed eyes? Answer yes or no."
            ),
            "head_turning": (
                "Using seizure semiology terminology, look for head version or "
                "head turning: the head visibly turns to one side during the "
                "event, including sustained or repeated turning. Does the "
                "video show head turning? Answer yes or no."
            ),
            "tonic": (
                "Using seizure semiology terminology, look for tonic motor "
                "activity: sustained stiffening, increased muscle tone, or "
                "rigid posturing of the body, trunk, face, or limbs. Does the "
                "video show tonic stiffening? Answer yes or no."
            ),
            "clonic": (
                "Using seizure semiology terminology, look for clonic motor "
                "activity: repeated rhythmic jerking movements of the face, "
                "trunk, arms, or legs. Does the video show clonic jerking? "
                "Answer yes or no."
            ),
            "full_body_shaking": (
                "Using seizure semiology terminology, look for generalized or "
                "widespread motor activity involving most of the body, such as "
                "whole-body shaking or repeated jerking rather than an isolated "
                "movement of one limb or facial region. Does the video show "
                "full-body shaking? Answer yes or no."
            ),
        },
    },
}

EXPERT_PROMPTS = {
    "arm_flexion": (
        "Does the patient flex their arms or arm at the elbows for at least a "
        "few video frames? Answer with 'yes' or 'no' and provide a "
        "justification for the answer. Respond with exactly one JSON object "
        "in the format { \"answer\": \"...\", \"justification\": \"...\" } "
        "and do not include any extra text outside of the JSON."
    ),
    "asynchronous_movement": (
        "Do you observe the patient's limbs shake with variable frequency or "
        "amplitude with respect to one another? Answer with 'yes' or 'no' and "
        "provide a justification for the answer. Respond with exactly one JSON "
        "object in the format { \"answer\": \"...\", \"justification\": "
        "\"...\" } and do not include any extra text outside of the JSON."
    ),
    "oral_automatisms": (
        "Does the patient exhibit repetitive, stereotyped mouth or tongue "
        "movements such as chewing, lip-smacking, or swallowing? Answer with "
        "'yes' or 'no' and provide a justification for the answer. Respond "
        "with exactly one JSON object in the format { \"answer\": \"...\", "
        "\"justification\": \"...\" } and do not include any extra text "
        "outside of the JSON."
    ),
    "blank_stare": (
        "Does the patient exhibit a blank stare? Answer with 'yes' or 'no' "
        "and provide a justification for the answer. Respond with exactly one "
        "JSON object in the format { \"answer\": \"...\", "
        "\"justification\": \"...\" } and do not include any extra text "
        "outside of the JSON."
    ),
    "close_eyes": (
        "Does the patient keep their eyes closed during the event? Answer with "
        "'yes' or 'no' and provide a justification for the answer. Respond "
        "with exactly one JSON object in the format { \"answer\": \"...\", "
        "\"justification\": \"...\" } and do not include any extra text "
        "outside of the JSON."
    ),
    "arm_straightening": (
        "Does the patient straighten or extend one or both arms at the elbow "
        "or shoulder during the event? Answer with 'yes' or 'no' and provide "
        "a justification for the answer. Respond with exactly one JSON object "
        "in the format { \"answer\": \"...\", \"justification\": \"...\" } "
        "and do not include any extra text outside of the JSON."
    ),
    "head_turning": (
        "Does the patient turn their head to one side during the event? "
        "Answer with 'yes' or 'no' and provide a justification for the "
        "answer. Respond with exactly one JSON object in the format "
        "{ \"answer\": \"...\", \"justification\": \"...\" } and do not "
        "include any extra text outside of the JSON."
    ),
    "tonic": (
        "Does the patient show sustained stiffening or tonic posturing of the "
        "body, face, trunk, or limbs during the event? Answer with 'yes' or "
        "'no' and provide a justification for the answer. Respond with "
        "exactly one JSON object in the format { \"answer\": \"...\", "
        "\"justification\": \"...\" } and do not include any extra text "
        "outside of the JSON."
    ),
    "clonic": (
        "Does the patient show repeated rhythmic jerking movements of the "
        "face, trunk, or limbs during the event? Answer with 'yes' or 'no' "
        "and provide a justification for the answer. Respond with exactly one "
        "JSON object in the format { \"answer\": \"...\", "
        "\"justification\": \"...\" } and do not include any extra text "
        "outside of the JSON."
    ),
    "full_body_shaking": (
        "Does the patient show shaking or jerking movements involving most of "
        "the body rather than only one isolated body part? Answer with 'yes' "
        "or 'no' and provide a justification for the answer. Respond with "
        "exactly one JSON object in the format { \"answer\": \"...\", "
        "\"justification\": \"...\" } and do not include any extra text "
        "outside of the JSON."
    ),
}

DEFAULT_PROMPT_TEMPLATES = (
    {
        "prompt_id": "minimal",
        "source": "minimal",
        "text": "Does this video show {feature_label}? Answer yes or no.",
    },
    {
        "prompt_id": "minimal_visible",
        "source": "paraphrase",
        "text": "Is {feature_label} visible in this video? Reply with yes or no.",
    },
    {
        "prompt_id": "minimal_classify",
        "source": "paraphrase",
        "text": "Classify whether the patient exhibits {feature_label}. Answer yes/no.",
    },
    {
        "prompt_id": "minimal_occurs",
        "source": "paraphrase",
        "text": (
            "Watch the video and decide whether {feature_label} occurs. "
            "Use only yes or no."
        ),
    },
    {
        "prompt_id": "brief_definition",
        "source": "non_expert_detail",
        "text": (
            "Does this video show {feature_label} ({feature_description})? "
            "Answer yes or no."
        ),
    },
)


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    feature: str
    source: str
    text: str


@dataclass(frozen=True)
class Example:
    example_id: str
    outer_fold: int
    split: str
    feature: str
    video_name: str
    video_path: str
    label: int
    unit: str
    segment_name: str | None = None
    patient_id: str | None = None


def normalize_label(value: str) -> int | None:
    text = (value or "").strip().lower()
    if text == "yes":
        return 1
    if text == "no":
        return 0
    return None


def label_text(label: int) -> str:
    return "yes" if label == 1 else "no"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_reflection_lm(value: str | None) -> str:
    model = value or os.environ.get("RQI_BEDROCK_MODEL") or "bedrock/minimax.minimax-m2.5"
    if "/" in model:
        return model
    return f"bedrock/{model}"


def load_annotations(csv_path: Path, name_column: str) -> dict[str, dict[str, str]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")
        if name_column not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} is missing '{name_column}'. "
                f"Available columns: {', '.join(reader.fieldnames)}"
            )
        annotations: dict[str, dict[str, str]] = {}
        for row in reader:
            video_name = (row.get(name_column) or "").strip()
            if video_name:
                annotations[video_name] = row
    return annotations


def load_folds(folds_path: Path) -> list[list[str]]:
    with folds_path.open() as handle:
        payload = json.load(handle)
    folds = payload.get("folds_videos")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"{folds_path} does not contain a non-empty folds_videos list")
    return [[str(video) for video in fold] for fold in folds]


def patient_id(video_name: str) -> str:
    match = re.match(r"^([A-Za-z]\d+)@", video_name)
    return match.group(1) if match else video_name


def parse_features(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_FEATURES
    return tuple(part.strip() for part in value.split(",") if part.strip())


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def feature_description(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, feature_label(feature))


def render_prompt(template: str, feature: str) -> str:
    return template.format(
        feature=feature,
        feature_label=feature_label(feature),
        feature_description=feature_description(feature),
    )


def seed_prompt_specs(
    feature: str,
    seed_prompt_ids: str | None,
    custom_seed_prompt: str | None,
) -> list[dict[str, str]]:
    if custom_seed_prompt:
        return [
            {
                "seed_prompt_id": "custom",
                "seed_prompt_source": "custom",
                "seed_prompt": render_prompt(custom_seed_prompt, feature),
            }
        ]

    prompt_ids = [
        part.strip()
        for part in (seed_prompt_ids or "minimal").split(",")
        if part.strip()
    ]
    specs: list[dict[str, str]] = []
    for prompt_id in prompt_ids:
        template = SEED_PROMPT_TEMPLATES.get(prompt_id)
        if template is None:
            raise ValueError(
                f"unknown seed prompt id {prompt_id!r}. "
                f"Available: {', '.join(sorted(SEED_PROMPT_TEMPLATES))}"
            )
        text_by_feature = template.get("text_by_feature")
        if isinstance(text_by_feature, dict) and feature in text_by_feature:
            raw_text = str(text_by_feature[feature])
        else:
            raw_text = str(template.get("text") or "")
        if not raw_text.strip():
            raise ValueError(f"empty seed prompt for {feature}/{prompt_id}")
        specs.append(
            {
                "seed_prompt_id": prompt_id,
                "seed_prompt_source": str(template.get("source") or prompt_id),
                "seed_prompt": render_prompt(raw_text, feature),
            }
        )
    return specs


def default_prompts(features: Iterable[str]) -> list[PromptSpec]:
    prompts: list[PromptSpec] = []
    for feature in features:
        if feature in EXPERT_PROMPTS:
            prompts.append(
                PromptSpec(
                    prompt_id="expert_clinician",
                    feature=feature,
                    source="expert",
                    text=EXPERT_PROMPTS[feature],
                )
            )
        for template in DEFAULT_PROMPT_TEMPLATES:
            prompts.append(
                PromptSpec(
                    prompt_id=str(template["prompt_id"]),
                    feature=feature,
                    source=str(template["source"]),
                    text=render_prompt(str(template["text"]), feature),
                )
            )
        for seed_spec in seed_prompt_specs(feature, "ilae_informed", None):
            prompts.append(
                PromptSpec(
                    prompt_id=seed_spec["seed_prompt_id"],
                    feature=feature,
                    source=seed_spec["seed_prompt_source"],
                    text=seed_spec["seed_prompt"],
                )
            )
    return prompts


def load_prompt_specs(path: Path | None, features: Iterable[str]) -> list[PromptSpec]:
    prompts = default_prompts(features)
    if path is None:
        return prompts

    with path.open() as handle:
        payload = json.load(handle)
    extra: list[PromptSpec] = []
    if isinstance(payload, dict):
        items = payload.items()
    elif isinstance(payload, list):
        items = [("*", payload)]
    else:
        raise ValueError("prompt JSON must be a dict or list")

    selected = set(features)
    for feature_key, raw_prompts in items:
        target_features = selected if feature_key == "*" else {str(feature_key)}
        if not isinstance(raw_prompts, list):
            raise ValueError(f"prompts for {feature_key!r} must be a list")
        for target_feature in target_features:
            for raw in raw_prompts:
                if isinstance(raw, str):
                    prompt_id = f"custom_{len(extra) + 1}"
                    source = "custom"
                    text = raw
                elif isinstance(raw, dict):
                    prompt_id = str(raw.get("prompt_id") or f"custom_{len(extra) + 1}")
                    source = str(raw.get("source") or "custom")
                    text = str(raw.get("text") or "")
                else:
                    raise ValueError("each prompt must be a string or object")
                if not text.strip():
                    raise ValueError(f"empty prompt text for {target_feature}/{prompt_id}")
                extra.append(
                    PromptSpec(
                        prompt_id=prompt_id,
                        feature=target_feature,
                        source=source,
                        text=render_prompt(text, target_feature),
                    )
                )
    return prompts + extra


def prompt_specs_to_rows(prompts: Iterable[PromptSpec]) -> list[dict[str, Any]]:
    return [
        {
            "prompt_id": prompt.prompt_id,
            "feature": prompt.feature,
            "source": prompt.source,
            "text": prompt.text,
        }
        for prompt in prompts
    ]


def resolve_video_path(
    data_root: Path,
    raw_video_root: Path | None,
    feature: str,
    video_name: str,
) -> Path:
    candidates = [
        data_root / feature / video_name,
        data_root / video_name,
    ]
    if raw_video_root is not None:
        candidates.extend(
            [
                raw_video_root / video_name,
                raw_video_root / feature / video_name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def derive_base_video_name(segment_name: str) -> str:
    stem, ext = os.path.splitext(segment_name)
    match = re.match(r"^(.*)_segment_\d+$", stem)
    if match:
        return f"{match.group(1)}{ext}"
    return segment_name


def build_segments_index(segments_root: Path) -> dict[str, list[Path]]:
    segments: dict[str, list[Path]] = defaultdict(list)
    if not segments_root.exists():
        return segments
    for path in segments_root.iterdir():
        if not path.is_file():
            continue
        base = derive_base_video_name(path.name)
        segments[base].append(path)
    for paths in segments.values():
        paths.sort(key=lambda item: item.name)
    return segments


def stratified_dev_split(
    videos: list[str],
    annotations: dict[str, dict[str, str]],
    feature: str,
    dev_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    by_label: dict[int, list[str]] = {0: [], 1: []}
    for video in videos:
        label = normalize_label(annotations[video].get(feature, ""))
        if label is not None:
            by_label[label].append(video)

    rng = random.Random(seed)
    train: set[str] = set()
    dev: set[str] = set()
    for label, group in by_label.items():
        group = list(group)
        rng.shuffle(group)
        if len(group) <= 1:
            train.update(group)
            continue
        dev_count = max(1, int(round(len(group) * dev_fraction)))
        dev_count = min(dev_count, len(group) - 1)
        dev.update(group[:dev_count])
        train.update(group[dev_count:])

    return train, dev


def balanced_example_subset(
    examples: list[dict[str, Any]],
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if not limit or len(examples) <= limit:
        return examples

    by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for example in examples:
        by_label[int(example["label"])].append(example)

    rng = random.Random(seed)
    for group in by_label.values():
        rng.shuffle(group)

    target_pos = min(len(by_label[1]), max(1, limit // 2))
    target_neg = min(len(by_label[0]), limit - target_pos)
    if target_pos + target_neg < limit:
        extra_pos = min(len(by_label[1]) - target_pos, limit - target_pos - target_neg)
        target_pos += extra_pos
    if target_pos + target_neg < limit:
        extra_neg = min(len(by_label[0]) - target_neg, limit - target_pos - target_neg)
        target_neg += extra_neg

    selected = by_label[1][:target_pos] + by_label[0][:target_neg]
    rng.shuffle(selected)
    return selected


def build_examples(
    *,
    annotations: dict[str, dict[str, str]],
    folds: list[list[str]],
    features: Iterable[str],
    data_root: Path,
    raw_video_root: Path | None,
    unit: str,
    segments_root: Path | None,
    dev_fraction: float,
    seed: int,
) -> list[Example]:
    segments_by_video = build_segments_index(segments_root) if segments_root else {}
    examples: list[Example] = []
    feature_set = tuple(features)
    for outer_idx, test_videos in enumerate(folds, start=1):
        test_set = set(test_videos)
        train_videos_all = [
            video
            for fold_idx, fold in enumerate(folds, start=1)
            if fold_idx != outer_idx
            for video in fold
            if video in annotations
        ]
        for feature in feature_set:
            labeled_train_videos = [
                video
                for video in train_videos_all
                if normalize_label(annotations[video].get(feature, "")) is not None
            ]
            train_set, dev_set = stratified_dev_split(
                labeled_train_videos,
                annotations,
                feature,
                dev_fraction,
                seed + outer_idx,
            )

            for split, split_videos in (
                ("train", sorted(train_set)),
                ("dev", sorted(dev_set)),
                ("test", sorted(test_set)),
            ):
                for video_name in split_videos:
                    if video_name not in annotations:
                        continue
                    label = normalize_label(annotations[video_name].get(feature, ""))
                    if label is None:
                        continue

                    if unit == "segment":
                        segment_paths = segments_by_video.get(video_name, [])
                        for segment_path in segment_paths:
                            examples.append(
                                Example(
                                    example_id=(
                                        f"fold{outer_idx}:{split}:{feature}:"
                                        f"{segment_path.stem}"
                                    ),
                                    outer_fold=outer_idx,
                                    split=split,
                                    feature=feature,
                                    video_name=video_name,
                                    video_path=str(segment_path),
                                    segment_name=segment_path.name,
                                    label=label,
                                    unit=unit,
                                    patient_id=patient_id(video_name),
                                )
                            )
                        continue

                    video_path = resolve_video_path(
                        data_root,
                        raw_video_root,
                        feature,
                        video_name,
                    )
                    examples.append(
                        Example(
                            example_id=f"fold{outer_idx}:{split}:{feature}:{video_name}",
                            outer_fold=outer_idx,
                            split=split,
                            feature=feature,
                            video_name=video_name,
                            video_path=str(video_path),
                            label=label,
                            unit=unit,
                            patient_id=patient_id(video_name),
                        )
                    )
    return examples


def example_to_row(example: Example) -> dict[str, Any]:
    row = {
        "example_id": example.example_id,
        "outer_fold": example.outer_fold,
        "split": example.split,
        "feature": example.feature,
        "video_name": example.video_name,
        "video_path": example.video_path,
        "label": example.label,
        "label_text": label_text(example.label),
        "unit": example.unit,
        "patient_id": example.patient_id,
    }
    if example.segment_name:
        row["segment_name"] = example.segment_name
    return row


def cmd_build_manifest(args: argparse.Namespace) -> None:
    features = parse_features(args.features)
    annotations = load_annotations(Path(args.csv), args.csv_name_column)
    folds = load_folds(Path(args.folds_json))
    examples = build_examples(
        annotations=annotations,
        folds=folds,
        features=features,
        data_root=Path(args.data_root),
        raw_video_root=Path(args.raw_video_root) if args.raw_video_root else None,
        unit=args.unit,
        segments_root=Path(args.segments_root) if args.segments_root else None,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )
    prompts = load_prompt_specs(Path(args.prompts_json) if args.prompts_json else None, features)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "examples.jsonl"
    prompts_path = output_dir / "prompts.jsonl"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(examples_path, [example_to_row(example) for example in examples])
    write_jsonl(prompts_path, prompt_specs_to_rows(prompts))

    counts: dict[str, int] = defaultdict(int)
    missing_paths: dict[str, int] = defaultdict(int)
    for example in examples:
        counts[f"{example.feature}/{example.split}"] += 1
        if not Path(example.video_path).exists():
            missing_paths[example.feature] += 1

    with manifest_path.open("w") as handle:
        json.dump(
            {
                "features": list(features),
                "folds": len(folds),
                "unit": args.unit,
                "dev_fraction": args.dev_fraction,
                "seed": args.seed,
                "examples_path": str(examples_path),
                "prompts_path": str(prompts_path),
                "counts": dict(sorted(counts.items())),
                "missing_video_paths": dict(sorted(missing_paths.items())),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"wrote {examples_path}")
    print(f"wrote {prompts_path}")
    print(f"wrote {manifest_path}")


def parse_answer(raw: str) -> int | None:
    text = raw.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        for key in ("prediction", "answer", "label", "pred_label"):
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int) and value in (0, 1):
                return value
            if isinstance(value, str):
                parsed = parse_answer(value)
                if parsed is not None:
                    return parsed
        return None

    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    tokens = normalized.split()
    if not tokens:
        return None
    if tokens[0] in {"yes", "y", "true", "positive", "present"}:
        return 1
    if tokens[0] in {"no", "n", "false", "negative", "absent"}:
        return 0
    if re.search(r"\banswer\s*:\s*yes\b", text, flags=re.IGNORECASE):
        return 1
    if re.search(r"\banswer\s*:\s*no\b", text, flags=re.IGNORECASE):
        return 0
    return None


def raw_response_text(stdout: str) -> str:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout.strip()
    if isinstance(payload, dict):
        for key in ("raw_response", "response", "text", "answer", "prediction"):
            if key in payload:
                return str(payload[key])
    return stdout.strip()


def normalize_backend_result(payload: Any, fallback_text: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        raw_response = ""
        for key in ("raw_response", "response", "text", "answer", "prediction"):
            if key in payload:
                raw_response = str(payload[key])
                break
        if not raw_response:
            raw_response = fallback_text

        pred = payload.get("pred_label")
        if isinstance(pred, bool):
            pred_label = int(pred)
        elif isinstance(pred, int) and pred in (0, 1):
            pred_label = pred
        else:
            pred_label = parse_answer(json.dumps(payload))
            if pred_label is None:
                pred_label = parse_answer(raw_response)

        return {
            "pred_label": pred_label,
            "raw_response": raw_response,
            "stdout": str(payload.get("stdout", fallback_text)),
            "stderr": str(payload.get("stderr", "")),
            "returncode": int(payload.get("returncode", 0) or 0),
        }

    text = str(payload if payload is not None else fallback_text)
    return {
        "pred_label": parse_answer(text),
        "raw_response": raw_response_text(text),
        "stdout": text,
        "stderr": "",
        "returncode": 0,
    }


def substitute_command(command: str, values: dict[str, str]) -> list[str]:
    rendered = command.format(**{key: shlex.quote(value) for key, value in values.items()})
    return shlex.split(rendered)


def run_backend_command(
    command: str,
    request: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    prompt_file_path: str | None = None
    command_values = {key: str(value) for key, value in request.items()}
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as prompt_file:
            prompt_file.write(str(request["prompt"]))
            prompt_file_path = prompt_file.name
        command_values["prompt_file"] = prompt_file_path

        if "{" in command and "}" in command:
            argv = substitute_command(command, command_values)
            input_payload = None
        else:
            argv = shlex.split(command)
            input_payload = json.dumps(request)

        completed = subprocess.run(
            argv,
            input=input_payload,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    finally:
        if prompt_file_path:
            try:
                os.unlink(prompt_file_path)
            except OSError:
                pass

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    try:
        payload: Any = json.loads(stdout)
    except json.JSONDecodeError:
        payload = stdout
    result = normalize_backend_result(payload, fallback_text=stdout)
    result["stdout"] = stdout
    result["stderr"] = stderr
    result["returncode"] = completed.returncode
    return result


class PersistentBackend:
    """Line-oriented JSONL backend process that keeps the MLLM loaded once."""

    def __init__(self, command: str, timeout: int) -> None:
        self.command = command
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "PersistentBackend":
        self.process = subprocess.Popen(
            shlex.split(self.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.process is None:
            return
        if self.process.stdin:
            try:
                self.process.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("persistent backend is not running")
        self.process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            return {
                "pred_label": None,
                "raw_response": "",
                "stdout": "",
                "stderr": "persistent backend exited without a response",
                "returncode": self.process.poll(),
            }
        try:
            payload: Any = json.loads(line)
        except json.JSONDecodeError:
            payload = line.strip()
        return normalize_backend_result(payload, fallback_text=line.strip())


def load_prompt_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(path)
    prompts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        prompts[(str(row["feature"]), str(row["prompt_id"]))] = row
    return prompts


def filter_examples(
    examples: list[dict[str, Any]],
    *,
    features: set[str] | None,
    folds: set[int] | None,
    splits: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = [
        example
        for example in examples
        if (features is None or str(example["feature"]) in features)
        and (folds is None or int(example["outer_fold"]) in folds)
        and (splits is None or str(example["split"]) in splits)
    ]
    if limit is not None:
        selected = selected[:limit]
    return selected


def existing_prediction_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for row in read_jsonl(path):
        keys.add(
            (
                str(row.get("example_id")),
                str(row.get("prompt_id")),
                str(row.get("feature")),
            )
        )
    return keys


def cmd_run(args: argparse.Namespace) -> None:
    if not args.backend_command and not args.persistent_backend_command:
        raise SystemExit("provide --backend-command or --persistent-backend-command")
    examples = read_jsonl(Path(args.examples_jsonl))
    prompts = load_prompt_rows(Path(args.prompts_jsonl))
    features = set(parse_features(args.features)) if args.features else None
    prompt_ids = (
        {part.strip() for part in args.prompt_ids.split(",") if part.strip()}
        if args.prompt_ids
        else None
    )
    folds = (
        {int(part.strip()) for part in args.folds.split(",") if part.strip()}
        if args.folds
        else None
    )
    splits = (
        {part.strip() for part in args.splits.split(",") if part.strip()}
        if args.splits
        else None
    )
    selected_examples = filter_examples(
        examples,
        features=features,
        folds=folds,
        splits=splits,
        limit=args.limit,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set() if args.overwrite else existing_prediction_keys(output_path)

    mode = "w" if args.overwrite else "a"
    total = 0
    skipped = 0
    backend_context = (
        PersistentBackend(args.persistent_backend_command, args.timeout)
        if args.persistent_backend_command
        else None
    )
    backend_cm = backend_context if backend_context is not None else None

    def run_request(request: dict[str, Any]) -> dict[str, Any]:
        if backend_context is not None:
            return backend_context.request(request)
        return run_backend_command(args.backend_command, request, args.timeout)

    with output_path.open(mode) as handle:
        if backend_cm is not None:
            backend_cm.__enter__()
        try:
            for example in selected_examples:
                feature = str(example["feature"])
                feature_prompts = [
                    prompt
                    for (prompt_feature, prompt_id), prompt in prompts.items()
                    if prompt_feature == feature
                    and (prompt_ids is None or prompt_id in prompt_ids)
                ]
                for prompt in sorted(feature_prompts, key=lambda row: str(row["prompt_id"])):
                    key = (
                        str(example["example_id"]),
                        str(prompt["prompt_id"]),
                        feature,
                    )
                    if key in existing:
                        skipped += 1
                        continue
                    request = {
                        "example_id": example["example_id"],
                        "outer_fold": example["outer_fold"],
                        "split": example["split"],
                        "feature": feature,
                        "feature_label": feature_label(feature),
                        "video_name": example["video_name"],
                        "video_path": example["video_path"],
                        "prompt_id": prompt["prompt_id"],
                        "prompt": prompt["text"],
                    }
                    result = run_request(request)
                    output = {
                        **request,
                        "label": example["label"],
                        "label_text": example["label_text"],
                        "prompt_source": prompt.get("source"),
                        **result,
                    }
                    handle.write(json.dumps(output, sort_keys=True) + "\n")
                    handle.flush()
                    total += 1
                    if total % args.progress_every == 0:
                        print(f"completed {total} calls", file=sys.stderr)
        finally:
            if backend_cm is not None:
                backend_cm.__exit__(None, None, None)
    print(f"wrote {total} new predictions to {output_path}")
    if skipped:
        print(f"skipped {skipped} existing predictions")


def safe_divide(num: float, den: float) -> float:
    return num / den if den else 0.0


def metrics_for_rows(rows: list[dict[str, Any]], unknown_policy: str) -> dict[str, Any]:
    tp = fp = tn = fn = unknown = 0
    for row in rows:
        label = int(row["label"])
        pred_raw = row.get("pred_label")
        pred = None if pred_raw is None else int(pred_raw)
        if pred not in (0, 1):
            unknown += 1
            if unknown_policy == "negative":
                pred = 0
            else:
                if label == 1:
                    fn += 1
                else:
                    fp += 1
                continue
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 1 and pred == 0:
            fn += 1

    support = tp + fp + tn + fn
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    accuracy = safe_divide(tp + tn, support)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "support": support,
        "positives": tp + fn,
        "negatives": tn + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "unknown": unknown,
    }


def group_by(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return groups


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def summarize_metrics(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    groups = group_by(metric_rows, ("feature", "split", "prompt_id"))
    for (feature, split, prompt_id), rows in sorted(groups.items()):
        f1_values = [float(row["f1"]) for row in rows]
        acc_values = [float(row["accuracy"]) for row in rows]
        summary_rows.append(
            {
                "feature": feature,
                "split": split,
                "prompt_id": prompt_id,
                "folds": len(rows),
                "f1_mean": mean(f1_values),
                "f1_sd": std(f1_values),
                "accuracy_mean": mean(acc_values),
                "accuracy_sd": std(acc_values),
                "support": sum(int(row["support"]) for row in rows),
                "unknown": sum(int(row["unknown"]) for row in rows),
            }
        )
    return summary_rows


def select_by_dev(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = group_by(metric_rows, ("feature", "outer_fold"))
    selected: list[dict[str, Any]] = []
    for (feature, outer_fold), rows in sorted(by_key.items()):
        dev_rows = [row for row in rows if row["split"] == "dev"]
        test_rows = [row for row in rows if row["split"] == "test"]
        if not dev_rows or not test_rows:
            continue
        best_dev = max(dev_rows, key=lambda row: (float(row["f1"]), float(row["accuracy"])))
        matching_test = [
            row for row in test_rows if row["prompt_id"] == best_dev["prompt_id"]
        ]
        for test_row in matching_test:
            selected.append(
                {
                    **test_row,
                    "selected_on": "dev",
                    "dev_f1": best_dev["f1"],
                    "dev_accuracy": best_dev["accuracy"],
                }
            )
    return selected


def cmd_score(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.predictions_jsonl))
    if args.splits:
        splits = {part.strip() for part in args.splits.split(",") if part.strip()}
        rows = [row for row in rows if row.get("split") in splits]
    metric_rows: list[dict[str, Any]] = []
    groups = group_by(rows, ("feature", "outer_fold", "split", "prompt_id"))
    for (feature, outer_fold, split, prompt_id), group_rows in sorted(groups.items()):
        metrics = metrics_for_rows(group_rows, args.unknown_policy)
        metric_rows.append(
            {
                "feature": feature,
                "outer_fold": outer_fold,
                "split": split,
                "prompt_id": prompt_id,
                **metrics,
            }
        )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "metrics_by_fold.csv", metric_rows)
    summary_rows = summarize_metrics(metric_rows)
    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    selected_rows = select_by_dev(metric_rows)
    write_csv(output_dir / "selected_by_dev_test_metrics.csv", selected_rows)
    selected_summary = summarize_metrics(selected_rows)
    write_csv(output_dir / "selected_by_dev_test_summary.csv", selected_summary)

    print(f"wrote {output_dir / 'metrics_by_fold.csv'}")
    print(f"wrote {output_dir / 'metrics_summary.csv'}")
    print(f"wrote {output_dir / 'selected_by_dev_test_metrics.csv'}")
    print(f"wrote {output_dir / 'selected_by_dev_test_summary.csv'}")


def evaluate_prompt_on_examples(
    prompt: str,
    examples: list[dict[str, Any]],
    command: str | None,
    timeout: int,
    backend_runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    prediction_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    total = len(examples)
    for idx, example in enumerate(examples, start=1):
        feature = str(example["feature"])
        request = {
            "example_id": example["example_id"],
            "outer_fold": example["outer_fold"],
            "split": example["split"],
            "feature": feature,
            "feature_label": feature_label(feature),
            "video_name": example["video_name"],
            "video_path": example["video_path"],
            "prompt_id": "gepa_candidate",
            "prompt": prompt,
        }
        cache_key = (prompt, str(example["example_id"]))
        if prediction_cache is not None and cache_key in prediction_cache:
            rows.append(dict(prediction_cache[cache_key]))
            print(
                f"[gepa-cache] {idx}/{total} feature={feature} "
                f"fold={example['outer_fold']} split={example['split']} "
                f"video={example['video_name']}",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(
            f"[gepa-eval] {idx}/{total} feature={feature} "
            f"fold={example['outer_fold']} split={example['split']} "
            f"video={example['video_name']}",
            file=sys.stderr,
            flush=True,
        )
        if backend_runner is not None:
            result = backend_runner(request)
        elif command:
            result = run_backend_command(command, request, timeout)
        else:
            raise ValueError("provide command or backend_runner")
        row = {**request, "label": example["label"], **result}
        if prediction_cache is not None:
            prediction_cache[cache_key] = dict(row)
        rows.append(row)
    score = metrics_for_rows(rows, unknown_policy="incorrect")["f1"]
    return float(score), rows


def compact_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_response = str(row.get("raw_response") or "")
    return {
        "video_name": row.get("video_name"),
        "split": row.get("split"),
        "label": label_text(int(row["label"])),
        "prediction": (
            label_text(int(row["pred_label"]))
            if row.get("pred_label") in (0, 1)
            else "unknown"
        ),
        "raw_response": raw_response[:240],
    }


def build_gepa_side_info(
    *,
    feature: str,
    prompt: str,
    score: float,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = metrics_for_rows(rows, unknown_policy="incorrect")
    false_negatives = [
        compact_prediction_row(row)
        for row in rows
        if int(row["label"]) == 1 and row.get("pred_label") != 1
    ][:8]
    false_positives = [
        compact_prediction_row(row)
        for row in rows
        if int(row["label"]) == 0 and row.get("pred_label") != 0
    ][:8]
    true_positives = [
        compact_prediction_row(row)
        for row in rows
        if int(row["label"]) == 1 and row.get("pred_label") == 1
    ][:4]
    true_negatives = [
        compact_prediction_row(row)
        for row in rows
        if int(row["label"]) == 0 and row.get("pred_label") == 0
    ][:4]
    return {
        "feature": feature,
        "feature_label": feature_label(feature),
        "feature_description": feature_description(feature),
        "current_prompt": prompt,
        "metric": "positive-class F1 for yes/no detection",
        "score": score,
        "confusion": {
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            "unknown": metrics["unknown"],
        },
        "failure_summary": (
            "False negatives are positive videos where the prompt caused the "
            "model to answer no or unknown. False positives are negative "
            "videos where the model answered yes. Improve the prompt to "
            "recover false negatives without creating false positives."
        ),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
    }


def cmd_optimize_gepa(args: argparse.Namespace) -> None:
    if not args.backend_command and not args.persistent_backend_command:
        raise SystemExit("provide --backend-command or --persistent-backend-command")
    load_env_file(Path(args.env_file))
    reflection_lm = resolve_reflection_lm(args.reflection_lm)
    try:
        from gepa.optimize_anything import GEPAConfig, EngineConfig, optimize_anything
        from gepa.optimize_anything import ReflectionConfig
    except Exception as exc:
        raise SystemExit(
            "GEPA/LiteLLM dependencies are not installed. Install gepa, litellm, "
            "and the reflection-model client dependencies before running "
            "`python prompt_robustness.py optimize-gepa ...`."
        ) from exc

    examples = read_jsonl(Path(args.examples_jsonl))
    features = set(parse_features(args.features)) if args.features else {
        str(example["feature"]) for example in examples
    }
    folds = (
        {int(part.strip()) for part in args.folds.split(",") if part.strip()}
        if args.folds
        else {int(example["outer_fold"]) for example in examples}
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimized_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def flush_partial_outputs() -> None:
        write_jsonl(output_dir / "gepa_optimized_prompts.partial.jsonl", optimized_rows)
        write_jsonl(output_dir / "gepa_predictions.partial.jsonl", prediction_rows)
        write_jsonl(output_dir / "gepa_candidate_history.partial.jsonl", candidate_rows)

    backend_context = (
        PersistentBackend(args.persistent_backend_command, args.timeout)
        if args.persistent_backend_command
        else None
    )
    if backend_context is not None:
        backend_context.__enter__()

    def run_request(request: dict[str, Any]) -> dict[str, Any]:
        if backend_context is not None:
            return backend_context.request(request)
        return run_backend_command(args.backend_command, request, args.timeout)

    try:
        for feature in sorted(features):
            seed_specs = seed_prompt_specs(
                feature,
                args.seed_prompt_ids,
                args.seed_prompt,
            )
            for seed_spec in seed_specs:
                seed_prompt = seed_spec["seed_prompt"]
                seed_prompt_id = seed_spec["seed_prompt_id"]
                for fold in sorted(folds):
                    train_examples = [
                        example
                        for example in examples
                        if example["feature"] == feature
                        and int(example["outer_fold"]) == fold
                        and example["split"] == "train"
                    ]
                    dev_examples = [
                        example
                        for example in examples
                        if example["feature"] == feature
                        and int(example["outer_fold"]) == fold
                        and example["split"] == "dev"
                    ]
                    test_examples = [
                        example
                        for example in examples
                        if example["feature"] == feature
                        and int(example["outer_fold"]) == fold
                        and example["split"] == "test"
                    ]
                    train_examples = balanced_example_subset(
                        train_examples,
                        args.max_train_examples,
                        seed=1000 * fold + sum(ord(char) for char in feature),
                    )
                    if not train_examples:
                        print(f"warning: no train examples for {feature} fold {fold}", file=sys.stderr)
                        continue

                    eval_counter = 0

                    def evaluator(candidate: str) -> tuple[float, dict[str, Any]]:
                        nonlocal eval_counter
                        eval_counter += 1
                        score, rows = evaluate_prompt_on_examples(
                            candidate,
                        train_examples,
                        args.backend_command,
                        args.timeout,
                        backend_runner=run_request,
                            prediction_cache=prediction_cache,
                        )
                        side_info = build_gepa_side_info(
                            feature=feature,
                            prompt=candidate,
                            score=score,
                            rows=rows,
                        )
                        metrics = side_info["confusion"]
                        candidate_rows.append(
                            {
                                "feature": feature,
                                "outer_fold": fold,
                                "seed_prompt_id": seed_prompt_id,
                                "seed_prompt_source": seed_spec["seed_prompt_source"],
                                "candidate_index": eval_counter,
                                "candidate_prompt": candidate,
                                "train_f1": score,
                                "train_examples": len(train_examples),
                                "tp": metrics["tp"],
                                "fp": metrics["fp"],
                                "tn": metrics["tn"],
                                "fn": metrics["fn"],
                                "unknown": metrics["unknown"],
                            }
                        )
                        write_jsonl(
                            output_dir / "gepa_candidate_history.partial.jsonl",
                            candidate_rows,
                        )
                        return score, side_info

                    result = optimize_anything(
                        seed_candidate=seed_prompt,
                        evaluator=evaluator,
                        objective=(
                            "Optimize a binary yes/no prompt for detecting the requested "
                            "seizure semiology feature in patient video. The prompt must "
                            "ask for only yes or no."
                        ),
                        config=GEPAConfig(
                            engine=EngineConfig(
                                max_metric_calls=args.max_metric_calls,
                                parallel=False,
                                max_workers=1,
                            ),
                            reflection=ReflectionConfig(
                                reflection_lm=reflection_lm,
                                reflection_minibatch_size=args.reflection_minibatch_size,
                            ),
                        ),
                    )
                    best_prompt = getattr(result, "best_candidate", None)
                    if isinstance(best_prompt, dict):
                        best_prompt_text = str(
                            best_prompt.get("system_prompt")
                            or best_prompt.get("prompt")
                            or next(iter(best_prompt.values()))
                        )
                    else:
                        best_prompt_text = str(best_prompt or seed_prompt)

                    dev_score, dev_rows = evaluate_prompt_on_examples(
                        best_prompt_text,
                        dev_examples,
                        args.backend_command,
                        args.timeout,
                        backend_runner=run_request,
                        prediction_cache=prediction_cache,
                    )
                    test_score, test_rows = evaluate_prompt_on_examples(
                        best_prompt_text,
                        test_examples,
                        args.backend_command,
                        args.timeout,
                        backend_runner=run_request,
                        prediction_cache=prediction_cache,
                    )
                    for row in dev_rows + test_rows:
                        prediction_rows.append(
                            {
                                **row,
                                "prompt_id": f"gepa_optimized_{seed_prompt_id}",
                                "prompt_source": "gepa",
                                "seed_prompt_id": seed_prompt_id,
                                "seed_prompt_source": seed_spec["seed_prompt_source"],
                            }
                        )
                    optimized_rows.append(
                        {
                            "feature": feature,
                            "outer_fold": fold,
                            "seed_prompt_id": seed_prompt_id,
                            "seed_prompt_source": seed_spec["seed_prompt_source"],
                            "seed_prompt": seed_prompt,
                            "optimized_prompt": best_prompt_text,
                            "train_examples": len(train_examples),
                            "dev_examples": len(dev_examples),
                            "test_examples": len(test_examples),
                            "dev_f1": dev_score,
                            "test_f1": test_score,
                        }
                    )
                    flush_partial_outputs()
                    print(
                        f"optimized {feature} fold {fold} seed={seed_prompt_id}: "
                        f"dev_f1={dev_score:.3f} test_f1={test_score:.3f}"
                    )
    finally:
        if backend_context is not None:
            backend_context.__exit__(None, None, None)

    write_jsonl(output_dir / "gepa_optimized_prompts.jsonl", optimized_rows)
    write_jsonl(output_dir / "gepa_predictions.jsonl", prediction_rows)
    write_jsonl(output_dir / "gepa_candidate_history.jsonl", candidate_rows)
    flush_partial_outputs()
    print(f"wrote {output_dir / 'gepa_optimized_prompts.jsonl'}")
    print(f"wrote {output_dir / 'gepa_predictions.jsonl'}")
    print(f"wrote {output_dir / 'gepa_candidate_history.jsonl'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and run prompt robustness experiments for video MLLMs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest")
    build.add_argument("--csv", default="evaluation/dataset/90_FeatureAnnotation.csv")
    build.add_argument("--csv-name-column", default="file_name")
    build.add_argument(
        "--folds-json",
        required=True,
        help="JSON file containing a folds_videos list from the CV split pipeline.",
    )
    build.add_argument(
        "--data-root",
        required=True,
        help="Directory containing videos, either by feature subdirectory or flat filename.",
    )
    build.add_argument(
        "--raw-video-root",
        default="/mnt/SSD1/linazhang/Dataset",
        help=(
            "Fallback directory containing one raw video file per CSV row. "
            "Use an empty string to disable."
        ),
    )
    build.add_argument("--segments-root", default=None)
    build.add_argument("--unit", choices=("video", "segment"), default="video")
    build.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    build.add_argument("--prompts-json", default=None)
    build.add_argument("--dev-fraction", type=float, default=0.25)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--output-dir", default="prompt_experiments")
    build.set_defaults(func=cmd_build_manifest)

    run = subparsers.add_parser("run")
    run.add_argument("--examples-jsonl", default="prompt_experiments/examples.jsonl")
    run.add_argument("--prompts-jsonl", default="prompt_experiments/prompts.jsonl")
    run.add_argument("--output-jsonl", default="prompt_experiments/predictions.jsonl")
    run.add_argument("--backend-command", default=None)
    run.add_argument(
        "--persistent-backend-command",
        default=None,
        help=(
            "Long-running JSONL backend command. The process should read one "
            "JSON request per stdin line and write one JSON response per stdout line."
        ),
    )
    run.add_argument("--features", default=None)
    run.add_argument("--prompt-ids", default=None)
    run.add_argument("--folds", default=None)
    run.add_argument("--splits", default="dev,test")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--timeout", type=int, default=300)
    run.add_argument("--progress-every", type=int, default=25)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(func=cmd_run)

    score = subparsers.add_parser("score")
    score.add_argument("--predictions-jsonl", default="prompt_experiments/predictions.jsonl")
    score.add_argument("--output-dir", default="prompt_experiments/scores")
    score.add_argument("--splits", default=None)
    score.add_argument(
        "--unknown-policy",
        choices=("negative", "incorrect"),
        default="incorrect",
    )
    score.set_defaults(func=cmd_score)

    gepa = subparsers.add_parser("optimize-gepa")
    gepa.add_argument("--examples-jsonl", default="prompt_experiments/examples.jsonl")
    gepa.add_argument("--output-dir", default="prompt_experiments/gepa")
    gepa.add_argument("--backend-command", default=None)
    gepa.add_argument(
        "--persistent-backend-command",
        default=None,
        help=(
            "Long-running JSONL backend command. Use this for local MLLMs so "
            "the model is loaded once across GEPA metric calls."
        ),
    )
    gepa.add_argument("--features", default=None)
    gepa.add_argument("--folds", default=None)
    gepa.add_argument(
        "--seed-prompt",
        default=None,
        help=(
            "Custom seed prompt template. If set, this overrides "
            "--seed-prompt-ids. Supports {feature_label} and "
            "{feature_description}."
        ),
    )
    gepa.add_argument(
        "--seed-prompt-ids",
        default="minimal",
        help=(
            "Comma-separated built-in seed prompts. Available: "
            f"{', '.join(sorted(SEED_PROMPT_TEMPLATES))}."
        ),
    )
    gepa.add_argument(
        "--reflection-lm",
        default=None,
        help=(
            "LiteLLM model string for GEPA reflection. Defaults to "
            "RQI_BEDROCK_MODEL from .env, or bedrock/minimax.minimax-m2.5. "
            "Plain Bedrock model IDs are automatically prefixed with bedrock/."
        ),
    )
    gepa.add_argument("--env-file", default=".env")
    gepa.add_argument("--max-metric-calls", type=int, default=12)
    gepa.add_argument("--reflection-minibatch-size", type=int, default=8)
    gepa.add_argument("--max-train-examples", type=int, default=8)
    gepa.add_argument("--timeout", type=int, default=300)
    gepa.set_defaults(func=cmd_optimize_gepa)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
