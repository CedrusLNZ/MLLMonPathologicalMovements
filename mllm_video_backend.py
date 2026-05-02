#!/usr/bin/env python3
"""Persistent JSONL video-MLLM backend for prompt robustness experiments.

The process reads one JSON request per stdin line and writes one JSON response
per stdout line. This keeps large video-MLLM weights loaded once while
`prompt_robustness.py` evaluates many prompt variants.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Protocol


class Backend(Protocol):
    def generate(self, video_path: str, prompt: str) -> str:
        ...


def parse_answer(raw: str) -> tuple[str | None, int | None]:
    text = (raw or "").strip()
    if not text:
        return None, None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        for key in ("answer", "prediction", "label", "pred_label"):
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, bool):
                return "yes" if value else "no", int(value)
            if isinstance(value, int) and value in (0, 1):
                return "yes" if value else "no", value
            if isinstance(value, str):
                return parse_answer(value)

    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    tokens = normalized.split()
    if not tokens:
        return None, None
    if tokens[0] in {"yes", "y", "true", "positive", "present"}:
        return "yes", 1
    if tokens[0] in {"no", "n", "false", "negative", "absent"}:
        return "no", 0
    if re.search(r"\banswer\s*:\s*yes\b", text, flags=re.IGNORECASE):
        return "yes", 1
    if re.search(r"\banswer\s*:\s*no\b", text, flags=re.IGNORECASE):
        return "no", 0
    if re.search(r"\byes\b", text, flags=re.IGNORECASE):
        return "yes", 1
    if re.search(r"\bno\b", text, flags=re.IGNORECASE):
        return "no", 0
    return None, None


@dataclass
class DummyBackend:
    answer: str = "no"

    def generate(self, video_path: str, prompt: str) -> str:
        return json.dumps({"answer": self.answer, "justification": "dummy backend"})


class QwenBackend:
    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: str,
        max_frames: int,
        fps: float,
        max_new_tokens: int,
        max_pixels: int,
        min_pixels: int,
        attn_implementation: str,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_frames = max_frames
        self.fps = fps
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.attn_implementation = attn_implementation
        self.model = None
        self.processor = None
        self.process_vision_info = None

    def _load(self) -> None:
        if self.model is not None:
            return

        hf_cache_dir = os.path.join(self.cache_dir, "huggingface")
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache_dir

        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "cache_dir": hf_cache_dir,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=hf_cache_dir,
        )
        self.process_vision_info = process_vision_info

    def generate(self, video_path: str, prompt: str) -> str:
        self._load()
        assert self.model is not None
        assert self.processor is not None
        assert self.process_vision_info is not None

        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": self.max_pixels,
                        "min_pixels": self.min_pixels,
                        "total_pixels": self.max_pixels * self.max_frames,
                        "fps": self.fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = self.process_vision_info(
            [messages],
            return_video_kwargs=True,
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=video_kwargs["fps"],
            padding=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return output_text[0]


class InternVLBackend:
    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: str,
        max_frames: int,
        max_new_tokens: int,
        tp: int,
        session_len: int,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_frames = max_frames
        self.max_new_tokens = max_new_tokens
        self.tp = tp
        self.session_len = session_len
        self.pipe = None
        self.GenerationConfig = None
        self.IMAGE_TOKEN = None
        self.encode_image_base64 = None

    def _load(self) -> None:
        if self.pipe is not None:
            return

        hf_cache_dir = os.path.join(self.cache_dir, "huggingface")
        lmdeploy_cache_dir = os.path.join(self.cache_dir, "lmdeploy")
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.makedirs(lmdeploy_cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        os.environ["HF_HUB_CACHE"] = hf_cache_dir
        os.environ["LMDEPLOY_CACHE_DIR"] = lmdeploy_cache_dir

        from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline
        from lmdeploy.vl.constants import IMAGE_TOKEN
        from lmdeploy.vl.utils import encode_image_base64

        engine_cfg = PytorchEngineConfig(tp=self.tp, session_len=self.session_len)
        self.pipe = pipeline(self.model_name, backend_config=engine_cfg)
        self.GenerationConfig = GenerationConfig
        self.IMAGE_TOKEN = IMAGE_TOKEN
        self.encode_image_base64 = encode_image_base64

    @staticmethod
    def _frame_indices(max_frame: int, fps: float, num_segments: int) -> Any:
        import numpy as np

        start_idx = 0
        end_idx = max_frame
        seg_size = float(end_idx - start_idx) / num_segments
        return np.array(
            [
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(num_segments)
            ]
        )

    def _load_video(self, video_path: str) -> list[Any]:
        from decord import VideoReader, cpu
        from PIL import Image

        if video_path.startswith("file://"):
            video_path = video_path[7:]

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        frame_indices = self._frame_indices(max_frame, fps, self.max_frames)
        return [
            Image.fromarray(vr[idx].asnumpy()).convert("RGB")
            for idx in frame_indices
        ]

    def generate(self, video_path: str, prompt: str) -> str:
        self._load()
        assert self.pipe is not None
        assert self.GenerationConfig is not None
        assert self.IMAGE_TOKEN is not None
        assert self.encode_image_base64 is not None

        pil_frames = self._load_video(video_path)
        question = (
            "".join(
                [f"Frame{i + 1}: {self.IMAGE_TOKEN}\n" for i in range(len(pil_frames))]
            )
            + prompt
        )
        content = [{"type": "text", "text": question}]
        for img in pil_frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "max_dynamic_patch": 1,
                        "url": (
                            "data:image/jpeg;base64,"
                            f"{self.encode_image_base64(img)}"
                        ),
                    },
                }
            )

        messages = [dict(role="user", content=content)]
        gen_cfg = self.GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        out = self.pipe(messages, gen_config=gen_cfg)
        return getattr(out, "text", str(out))


def build_backend(args: argparse.Namespace) -> Backend:
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.backend == "dummy":
        return DummyBackend(answer=args.dummy_answer)

    if args.backend == "qwen":
        model_name = args.model_name or "Qwen/Qwen2.5-VL-32B-Instruct"
        return QwenBackend(
            model_name=model_name,
            cache_dir=args.cache_dir,
            max_frames=args.max_frames,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
            max_pixels=args.qwen_max_pixels,
            min_pixels=args.qwen_min_pixels,
            attn_implementation=args.qwen_attn_implementation,
        )

    if args.backend == "internvl":
        model_name = args.model_name or "OpenGVLab/InternVL3_5-38B"
        return InternVLBackend(
            model_name=model_name,
            cache_dir=args.cache_dir,
            max_frames=args.max_frames,
            max_new_tokens=args.max_new_tokens,
            tp=args.tp,
            session_len=args.session_len,
        )

    raise ValueError(f"unknown backend: {args.backend}")


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def handle_request(backend: Backend, request: dict[str, Any]) -> dict[str, Any]:
    video_path = str(request["video_path"])
    prompt = str(request["prompt"])
    try:
        raw_response = backend.generate(video_path, prompt)
        answer, pred_label = parse_answer(raw_response)
        return {
            "example_id": request.get("example_id"),
            "prompt_id": request.get("prompt_id"),
            "feature": request.get("feature"),
            "answer": answer,
            "pred_label": pred_label,
            "raw_response": raw_response,
        }
    finally:
        clear_cuda_cache()


def serve(args: argparse.Namespace) -> None:
    backend = build_backend(args)
    print(f"loaded backend config: {args}", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("shutdown"):
                break
            with contextlib.redirect_stdout(sys.stderr):
                response = handle_request(backend, request)
        except Exception as exc:
            clear_cuda_cache()
            response = {
                "answer": None,
                "pred_label": None,
                "raw_response": "",
                "stderr": repr(exc),
                "returncode": 1,
            }
        print(json.dumps(response, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent video-MLLM backend for prompt_robustness.py"
    )
    parser.add_argument(
        "--backend",
        choices=("dummy", "qwen", "internvl"),
        default="dummy",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--cache-dir", default="./mllm_cache")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)

    parser.add_argument("--qwen-max-pixels", type=int, default=602112)
    parser.add_argument("--qwen-min-pixels", type=int, default=16 * 28 * 28)
    parser.add_argument(
        "--qwen-attn-implementation",
        default="flash_attention_2",
        help="Set to an empty string to use the transformers default.",
    )

    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--session-len", type=int, default=32768)

    parser.add_argument("--dummy-answer", choices=("yes", "no"), default="no")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
