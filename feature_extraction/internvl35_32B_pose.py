# ExtractFeature_internvl3_5.py
# LMDeploy + InternVL3.5-8B (PyTorch backend; IMAGE_TOKEN + base64 data URL)
# - Output CSV: file_name + for each feature two columns [feature, justification_for_feature]
# - Per-video logs optional (disabled by default)

import os
import re
import csv
import json
import time
import argparse
import traceback
from tqdm import tqdm

import numpy as np
import pandas as pd
import requests
from PIL import Image
from decord import VideoReader, cpu

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64


# --------------------------- CLI ---------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Seizure Video Feature Extraction using InternVL3.5-8B via LMDeploy')

    # GPU settings
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID(s) to use (default: 0)')

    # Tensor parallelism settings
    parser.add_argument('--tp', type=int, default=1, 
                        help='Tensor parallel degree (default: 1)')

    # Model settings
    parser.add_argument('--model_name', type=str, default='OpenGVLab/InternVL3_5-8B',
                        help='Model name to use (default: OpenGVLab/InternVL3_5-8B)')

    # Data settings
    parser.add_argument('--dataset_dir', type=str,
                        default='/mnt/SSD3/lina/ucla2/pose_segments',
                        help='Directory containing pose segment videos')
    
    parser.add_argument('--max_frames', type=int, default=60,
                        help='Maximum frames sampled per video (segment-center sampling)')

    # Output settings
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results CSV and optional logs')
    # parser.add_argument('--max_videos', type=int, default=-1,
    #                     help='Max number of videos to process; -1 for all')

    # Cache settings
    parser.add_argument('--cache_dir', type=str, default='./model_cache/',
                        help='Directory for model caches')

    # Processing settings
    parser.add_argument('--max_retries', type=int, default=10,
                        help='Max retries for a feature prompt on errors')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help='Max new tokens for generation')

    # Video range settings
    parser.add_argument('--videos_range', type=str, default='1-2314',
                       help='Range of videos to process (e.g., "0,9" for first 10 videos, "10,19" for next 10 videos, etc.)')
                        

    # Logging settings
    parser.add_argument('--disable_logs', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True,
                        help='Disable per-video log CSVs when True (default: True)')

    return parser.parse_args()


args = parse_arguments()

# --------------------------- Environment ---------------------------

# GPU visibility
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
videos_range = args.videos_range

# Caches
hf_cache_dir = os.path.join(args.cache_dir, 'huggingface')
lmdeploy_cache_dir = os.path.join(args.cache_dir, 'lmdeploy')
os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(lmdeploy_cache_dir, exist_ok=True)

os.environ['HF_HOME'] = hf_cache_dir
os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
os.environ['HF_HUB_CACHE'] = hf_cache_dir
os.environ['LMDEPLOY_CACHE_DIR'] = lmdeploy_cache_dir

# Paths
model_name = args.model_name
dataset_dir = args.dataset_dir
inference_dir = args.output_dir
os.makedirs(inference_dir, exist_ok=True)
if not args.disable_logs:
    inference_log_dir = os.path.join(inference_dir, 'log')
    os.makedirs(inference_log_dir, exist_ok=True)

# Output CSV path follows Qwen naming
inf_result_csv_fp = os.path.join(inference_dir, f'Task1_{model_name.split("/")[-1]}_{videos_range}_pose.csv')

# --------------------------- Features ---------------------------

all_features = [
    'arm_flexion', 'arms_move_simultaneously', 'occur_during_sleep',
    'tonic', 'clonic', 'arm_straightening', 'figure4',
    'limb_automatisms', 'asynchronous_movement', 'pelvic_thrusting',
    'full_body_shaking'
]

MAX_FRAMES = args.max_frames

# --------------------------- Video I/O ---------------------------

def download_video(url, dest_path):
    """Optional helper to download remote videos."""
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """
    Compute segment-center indices between [start, end] with 'num_segments' segments.
    If 'bound' is None, cover the whole video.
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, num_segments=32):
    """
    Read frames using decord at original resolution (no resize).
    Return a list of PIL.Image.
    """
    # Support file:// prefix
    if video_path.startswith('file://'):
        video_path = video_path[7:]

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    imgs = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        imgs.append(img)
    return imgs

# --------------------------- LMDeploy (PyTorch backend) ---------------------------

# Keep tp=1 in code (do not add new CLI arg unless you purposely support tensor-parallel)
# engine_cfg = PytorchEngineConfig(
#     tp=1,
#     session_len=32768,  # common setting; larger context => more VRAM
# )

engine_cfg = PytorchEngineConfig(tp=args.tp, session_len=32768)
pipe = pipeline(model_name, backend_config=engine_cfg)


def inference(video_path, prompt, max_new_tokens=None, temperature=0.0, do_sample=False):
    """
    InternVL3.5-8B inference via LMDeploy:
      - Sample MAX_FRAMES frames at segment centers
      - Construct question text with IMAGE_TOKEN markers
      - Send text first, then all frames as data URLs (base64)
    """
    if max_new_tokens is None:
        max_new_tokens = args.max_new_tokens

    pil_frames = load_video(video_path, bound=None, num_segments=MAX_FRAMES)
    question = "".join([f"Frame{i+1}: {IMAGE_TOKEN}\n" for i in range(len(pil_frames))]) + prompt

    content = [{'type': 'text', 'text': question}]
    for img in pil_frames:
        content.append({
            'type': 'image_url',
            'image_url': {
                'max_dynamic_patch': 1,  # tunable
                'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'
            }
        })

    messages = [dict(role='user', content=content)]
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample
    )
    out = pipe(messages, gen_config=gen_cfg)
    return getattr(out, 'text', str(out))

# --------------------------- Prompts ---------------------------

def get_prompts():
    """
    Build per-feature prompts. Each prompt asks for a single JSON object:
    { "answer": "...", "justification": "..." }.
    """
    feature_names = all_features.copy()
    prompts = {}
    for feature in feature_names:
        if feature == 'blank_stare':
            prompts[feature] = "Does the patient exhibit a blank stare? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'arm_flexion':
            prompts[feature] = "Does the patient flex their arms or arm at the elbows for at least a few video frames? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'arms_move_simultaneously':
            prompts[feature] = "Do the patient's arms start moving approximately at the same time? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        # if feature == 'gender':
        #     prompts[feature] = "Please identify the gender of the patient in the video. Please answer with \"female\" or \"male\". Do not include extra text in your output—only the answer."
        if feature == 'occur_during_sleep':
            prompts[feature] = "Is the patient sleeping at the beginning of the video? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        # if feature == 'ictal_vocalization':
        #     prompts[feature] = "Does the patient make any groaning, moaning, guttural sounds or do they utter stereotyped repetitive phrases? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'close_eyes':
            prompts[feature] = "Do the patient's eyes remain consistently closed or mostly closed throughout the video? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'eye_blinking':
            prompts[feature] = "Does the patient show rapid blinking of the eyes during the video? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'tonic':
            prompts[feature] = "The tonic phase is marked by a sudden onset of sustained stiffness or rigidity, usually lasting 5-20 seconds. This stiffness may be generalized, with all limbs held in fixed extension or flexion posture and can include stiffening of the head and axial body. It may also be focal involving a subset of body parts or just one body part at a time. Does this patient show tonic? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer. "
        if feature == 'clonic':
            prompts[feature] = "Clonic Phase: Rhythmic, jerking muscle contractions involving the limbs, face, and trunk. The jerking movements gradually slow before ceasing entirely. Clonic movements present as rhythmic, regular stereotyped contraction and relaxation of the affected body parts. Does this patient show clonic? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'arm_straightening':
            prompts[feature] = "Does the patient straighten or extend their arms or arm at the elbow for at least a few video frames? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'figure4':
            prompts[feature] = "Figure 4 refers to a tonic sustained posture where one arm is flexed while the other is extended at the same time. Does the patient exhibit a Figure 4 posture? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'oral_automatisms':
            prompts[feature] = "Does the patient exhibit repetitive, stereotyped mouth or tongue movements such as chewing, lip-smacking, or swallowing? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'limb_automatisms':
            prompts[feature] = "Does the patient exhibit repetitive, stereotyped limb movements such as fumbling, picking, rubbing or patting?  Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'face_pulling':
            prompts[feature] = "Does the patient exhibit unilateral sustained face-pulling movements? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'face_twitching':
            prompts[feature] = "Are there small muscle twitches observed on the patient's face? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'head_turning':
            prompts[feature] = "Does the patient forcibly or stiffly rotate their head to one side in the video? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'asynchronous_movement':
            prompts[feature] = "Do you observe the patient's limbs shake with variable frequency or amplitude with respect to one another? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        if feature == 'pelvic_thrusting':
            prompts[feature] = "Does the patient display repetitive, rhythmic, anteroposterior (forward-and-backward) movements of the hips? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer. "
        if feature == 'full_body_shaking':
            prompts[feature] = "Does the patient experience shaking of the entire body including arms, legs, torso? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."

    assert len(feature_names) == len(prompts), f"feature_names length {len(feature_names)} and prompt_list lengths {len(prompts)} does not match."
    return prompts

# --------------------------- CSV helpers ---------------------------

def append_to_csv(csv_file, data, header=None):
    """
    Append a row to a CSV file. If it does not exist and 'header' is provided,
    write the header first.
    """
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if (not file_exists) and (header is not None):
            w.writerow(header)
        w.writerow(data)

# --------------------------- Robust JSON parsing ---------------------------

def parse_answer_json(raw_text):
    """
    Parse only {answer, justification} from model output.
    Robust pipeline:
      1) strict JSON parse after sanitization
      2) soft regex parse
      3) minimal fallback (infer yes/no/na; empty justification)

    Additionally:
      - If `justification` is itself JSON-like, try to parse it again and extract
        'justification' (or 'reason'/'explanation'/'rationale').
      - Strip outer quotes (", ', “ ”, ‘ ’) and unescape common patterns to avoid quotes wrapping.
    """
    import json as _json
    import re as _re

    def _sanitize_top(s: str) -> str:
        s = (s if isinstance(s, str) else str(s)).strip()
        s = _re.sub(r"^```[a-zA-Z]*\s*", "", s)   # remove leading code fence
        s = _re.sub(r"\s*```$", "", s)           # remove trailing code fence
        m = _re.search(r'\{.*\}', s, flags=_re.S)
        js = m.group(0) if m else s               # take first {...} if present
        # common fixes
        js = _re.sub(r"(?<!\\)'", '"', js)        # naive: single -> double quotes
        js = _re.sub(r",\s*([}\]])", r"\1", js)   # trailing commas
        js = js.replace("None", '"N/A"').replace("True", '"yes"').replace("False", '"no"')
        return js

    def _norm_ans(a) -> str:
        a = (str(a).strip().lower() if a is not None else '')
        if a in ('yes', 'no', 'na'):
            return a
        if 'yes' in a: return 'yes'
        if 'no'  in a: return 'no'
        if 'na'  in a: return 'na'
        return 'fail'

    def _strip_outer_quotes(t: str) -> str:
        """Remove symmetric outer quotes repeatedly (", ', “”, ‘’)."""
        if not t:
            return t
        t = t.strip()
        pairs = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")]
        for _ in range(3):
            changed = False
            for lq, rq in pairs:
                if t.startswith(lq) and t.endswith(rq) and len(t) >= 2:
                    t = t[len(lq):-len(rq)].strip()
                    changed = True
            if not changed:
                break
        return t

    def _unescape_and_unwrap(t) -> str:
        t = "" if t is None else str(t)
        # unescape common patterns
        t = t.replace(r'\"', '"').replace(r"\'", "'")
        t = t.replace("&quot;", '"').replace("&#34;", '"').replace("&#39;", "'")
        # strip outer quotes (ascii + unicode)
        t = _strip_outer_quotes(t)
        return t

    def _extract_nested_just(j):
        """If `j` looks like JSON, parse and extract its justification-like field; otherwise clean quotes."""
        if not isinstance(j, str):
            return j
        t = j.strip()
        if not t:
            return ""
        looks_json = (t.startswith('{') and t.endswith('}')) or ('"answer"' in t and '"justification"' in t)
        if looks_json:
            try:
                inner = _json.loads(_sanitize_top(t))
                if isinstance(inner, dict):
                    for key in ('justification', 'reason', 'explanation', 'rationale'):
                        if key in inner:
                            v = inner[key]
                            return _unescape_and_unwrap(v if isinstance(v, str) else _json.dumps(v, ensure_ascii=False))
                return _unescape_and_unwrap(_json.dumps(inner, ensure_ascii=False))
            except Exception:
                m = _re.search(r'"justification"\s*:\s*"(.*?)"', t, flags=_re.S | _re.I)
                if m:
                    return _unescape_and_unwrap(m.group(1))
        return _unescape_and_unwrap(t)

    def _first_answer_from_free_text(s: str) -> str:
        s_low = s.lower()
        if _re.search(r'\bna\b', s_low): return 'na'
        if _re.search(r'\byes\b', s_low): return 'yes'
        if _re.search(r'\bno\b',  s_low): return 'no'
        return 'fail'

    s = raw_text if isinstance(raw_text, str) else str(raw_text)

    # 1) Strict JSON parse
    try:
        js = _sanitize_top(s)
        obj = _json.loads(js)
        ans = _norm_ans(obj.get('answer'))
        just = obj.get('justification', '')
        just = _extract_nested_just(just)
        return {'answer': ans or 'fail'}
    except Exception:
        pass

    # 2) Soft regex parse
    try:
        js = _sanitize_top(s)
        ans_m  = re.search(r'"answer"\s*:\s*"(yes|no|na)"', js, flags=re.I)
        just_m = re.search(r'"justification"\s*:\s*"(.*?)"', js, flags=re.I | re.S)
        ans = _norm_ans(ans_m.group(1) if ans_m else '')
        just = _extract_nested_just(just_m.group(1) if just_m else '')
        if not ans:
            ans = _first_answer_from_free_text(js)
        return {'answer': ans, 'justification': just}
    except Exception:
        pass

    # 3) Minimal fallback
    return {'answer': _first_answer_from_free_text(s), 'justification': ''}

# --------------------------- Core loop ---------------------------

def ExtractFeatureByVLM(video_path, file_name, video_idx_info, log_csv, prompt_dict):
    """
    For each feature, run inference and parse the JSON answer.
    Log (file_name, prompt, answer, justification) into a per-video CSV if logging is enabled.
    """
    answer_dict = {}

    for feature in tqdm(prompt_dict.keys(),
                        desc=f"Inferencing on video[{video_idx_info[0]}/{video_idx_info[1]}]",
                        total=len(prompt_dict.keys())):

        prompt = prompt_dict[feature]
        max_retries = args.max_retries
        raw_answer = ""

        for retry_count in range(max_retries):
            try:
                raw_answer = inference(video_path, prompt, max_new_tokens=args.max_new_tokens, temperature=0.0, do_sample=False)

            except Exception as e:
                print(f"Error in prompt for feature: {feature}: {prompt}")
                print(f"Raw VLM response: {raw_answer}")
                print(f"Exception: {str(e)}. Retrying ({retry_count + 1}/{max_retries})...")
                traceback.print_exc()

        else:
            # Retries exhausted
            answer_dict[feature] = {'answer': "fail"}
            if not args.disable_logs and log_csv is not None:
                append_to_csv(log_csv, [file_name, prompt, "fail", "fail"])

    return raw_answer

# --------------------------- Main ---------------------------

def main():
    prompts = get_prompts()

    # CSV header: file_name + for each feature: [feature, justification_for_feature]
    output_header = ['file_name']
    for feat in prompts.keys():
        output_header.append(feat)

    # Prepare result CSV (write header if new)
    if os.path.exists(inf_result_csv_fp):
        try:
            result_df = pd.read_csv(inf_result_csv_fp)
            processed = set(result_df.iloc[:, 0].tolist())
        except Exception:
            processed = set()
    else:
        processed = set()
        with open(inf_result_csv_fp, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(output_header)

    # List dataset videos
    input_videos_files = sorted(list(set(os.listdir(dataset_dir))))
    # if args.max_videos == -1:
    #     video_list = input_videos_files
    # else:
    #     video_list = input_videos_files[:args.max_videos]

        # make sure videos_range is valid
    videos_range = args.videos_range.split('-')
    videos_range = [int(videos_range[0]), int(videos_range[1])]
    if len(videos_range) != 2:
        raise ValueError("videos_range must be a comma-separated string of two numbers")
    if int(videos_range[0]) > int(videos_range[1]):
        raise ValueError("videos_range[0] must be less than videos_range[1]")
    if int(videos_range[0]) < 0:
        videos_range[0] = 1
        # add warning
        print(f"Warning: videos_range[0] is less than 0, set to 0")
    if int(videos_range[1]) > len(input_videos_files):
        videos_range[1] = len(input_videos_files)
        # add warning
        print(f"Warning: videos_range[1] is greater than the number of videos, set to {len(input_videos_files)}")
    video_list = input_videos_files[(videos_range[0]-1) : (videos_range[1])]    



    # Iterate videos
    for video_idx, file_name in enumerate(video_list, start=1):
        if file_name in processed:
            print(file_name, "already processed")
            continue

        video_path = os.path.join(dataset_dir, file_name)
        row_to_write = [file_name]

        # Optional per-video log file
        log_file = None
        if not args.disable_logs:
            log_file = os.path.join(inference_log_dir, f'{file_name}---log.csv')
            with open(log_file, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(["file_name", "prompt", "answer", "justification"])

        if not os.path.exists(video_path):
            # File missing: write placeholders
            for _ in prompts.keys():
                row_to_write.append("VideoNotExist")
            append_to_csv(inf_result_csv_fp, row_to_write)
            continue

        print(f"Processing: {video_path}")
        try:
            answer_dict = ExtractFeatureByVLM(
                video_path, file_name,
                (video_idx, len(video_list)),
                log_file, prompts
            )
            for feat in prompts.keys():
                data = answer_dict.get(feat, None)
                if not data:
                    row_to_write.extend(["fail", "fail"])
                else:
                    row_to_write.extend([data.get('answer', 'fail'),
                                         data.get('justification', 'fail')])

        except Exception as e:
            print(f"Error processing video {file_name}: {str(e)}")
            traceback.print_exc()
            for _ in prompts.keys():
                row_to_write.append("fail")

        append_to_csv(inf_result_csv_fp, row_to_write)

    if args.disable_logs:
        print(f"Processing complete. Results: '{inf_result_csv_fp}'. Log files are disabled.")
    else:
        print(f"Processing complete. Results: '{inf_result_csv_fp}'. Logs under '{inference_log_dir}'.")


if __name__ == "__main__":
    print("Starting seizure video feature extraction with InternVL3.5-8B (LMDeploy)...")
    print(args)
    # print(f"GPU: {args.gpu}")
    # print(f"Model: {args.model_name}")
    # print(f"Dataset: {args.dataset_dir}")
    # print(f"Output: {args.output_dir}")
    # #print(f"Max frames: {args.max_frames}")
    # print(f"videos_range: {args.videos_range}")
    # print(f"Log files: {'Disabled' if args.disable_logs else 'Enabled'}")
    print("-" * 50)
    main()


