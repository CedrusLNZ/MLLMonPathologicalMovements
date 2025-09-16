# import torch
# import torchaudio
# from huggingface_hub import snapshot_download
# from llava.entry import load as af3_load
# from llava.media import Sound as AF3Sound
# import numpy as np
# from tqdm import tqdm

import os, re, csv, json, argparse, tempfile, sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from huggingface_hub import snapshot_download

# -------------------- Args --------------------
def parse_args():
    p = argparse.ArgumentParser("ALM Task1 (AF3 · one-pass per audio)")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--dataset_dir", type=str, default="/mnt/SSD1/prateik/seizure_audition/data/audio")
    p.add_argument("--output_dir", type=str, default=os.path.expanduser("~/alm_results_full"))
    p.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/hf_cache"))
    p.add_argument("--disable_logs", type=lambda x: x.lower() in ("1","true","yes"), default=True)
    p.add_argument("--max_groups", type=int, default=-1, help="process first N files; -1 = all")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--model_repo", type=str, default="nvidia/audio-flamingo-3")
    p.add_argument("--max_retries", type=int, default=2)
    return p.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------- Speed-safe knobs --------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# -------------------- Caches --------------------
hf_root  = os.path.join(args.cache_dir)
hf_cache = os.path.join(hf_root, "huggingface")
os.makedirs(hf_cache, exist_ok=True)
os.environ["HF_HOME"]            = hf_cache
os.environ["TRANSFORMERS_CACHE"] = hf_cache
os.environ["HF_HUB_CACHE"]       = hf_cache

os.makedirs(args.output_dir, exist_ok=True)
if not args.disable_logs:
    os.makedirs(os.path.join(args.output_dir, "log"), exist_ok=True)

# -------------------- Prompts --------------------
PROMPTS: Dict[str, str] = {
    "verbal_responsiveness":
        "Based on the audio, does it contain a clear verbal interaction or conversation where the doctor addresses the patient, and the patient responds coherently? "       
        "Answer 'yes' if a coherent verbal response is detected following an address. "
        "Answer 'no' if an address is detected but the reply is absent, unclear, or non-verbal. "
        "Answer 'NA' if no clear verbal address to the patient is detected in the audio. "
        "Respond with exactly one JSON object in the format "
        "{\"answer\":\"yes|no|NA\", \"justification\":\"brief explanation of the evidence you heard (quote key words or describe silence/non-verbal sounds)\"} "
        "and do not include any extra text outside the JSON.",

    "ictal_vocalization":
        "Does the patient make any groaning, moaning, guttural sounds or do they utter stereotyped repetitive phrases? "
        "Answer with 'yes' or 'no' and provide a justification for the answer. "
        "Respond with exactly one JSON object in the format {\"answer\":\"yes|no\", "
        "\"justification\":\"brief explanation\", \"start_time\":\"MM:SS\" or \"N/A\"} and do not include any extra text outside of the JSON."
}

CSV_HEADER = [
    "file_name",
    "verbal_responsiveness", "justification_for_verbal_responsiveness",
    "ictal_vocalization", "justification_for_ictal_vocalization",
]
CSV_PATH = os.path.join(args.output_dir, "AF3_93_audio_clear_results.csv")

# -------------------- Utils --------------------
def save_row(csv_file: str, row: List[str], header: List[str] = None):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if (not file_exists) and (header is not None):
            w.writerow(header)
        w.writerow(row)

def load_audio_to_16k_mono(path: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(path)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        sr = target_sr
    return wav.squeeze(0).cpu().numpy().astype(np.float32), sr

# -- strict JSON clean --
def clean_json_response(raw_response: str):
    if not raw_response:
        return None
    s = raw_response.strip()
    start = s.find("{"); end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    frag = s[start:end+1]
    frag = frag.replace("'", '"')
    frag = re.sub(r",\s*([}\]])", r"\1", frag)
    try:
        return json.loads(frag)
    except json.JSONDecodeError:
        try:
            frag2 = "".join(ch for ch in frag if ord(ch) < 128)
            frag2 = re.sub(r'([{,])\s*([A-Za-z0-9_]+)\s*:', r'\1 "\2":', frag2)
            frag2 = re.sub(r",\s*([}\]])", r"\1", frag2)
            return json.loads(frag2)
        except Exception:
            return None

def strict_json_load(raw: str):
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    return clean_json_response(raw)

def norm_answer(a: str) -> str:
    if not isinstance(a, str):
        a = str(a or "")
    a = a.strip().lower()
    if a in ("yes", "no", "na", "n/a"):
        return "na" if a == "n/a" else a
    return "fail"

def tolerant_parse_from_text(raw: str, feature_key: str) -> Dict[str, str] | None:
    if not isinstance(raw, str):
        raw = str(raw or "")
    s = raw.strip()
    if not s:
        return None
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    m = re.match(r'^\s*(yes|no)\b[,:;]?\s*(.*)$', s, flags=re.I|re.S)
    if m:
        ans = norm_answer(m.group(1))
        just = (m.group(2) or "").strip().strip('"')
        return {"answer": ans, "justification": just}

    low = s.lower()
    if low in ("na", "n/a"):
        if feature_key == "verbal_responsiveness":
            return {"answer": "na", "justification": "no clear verbal address is heard in the audio, so verbal responsiveness cannot be determined."}
        else:
            return {"answer": "na", "justification": "the audio lacks sufficient cues to determine this feature."}

    if re.search(r'\byes\b', low):
        return {"answer": "yes", "justification": s[:500]}
    if re.search(r'\bno\b', low):
        return {"answer": "no", "justification": s[:500]}
    if re.search(r'\bna\b|\bn\/a\b', low):
        if feature_key == "verbal_responsiveness":
            return {"answer": "na", "justification": "no clear verbal address is heard in the audio, so verbal responsiveness cannot be determined."}
        return {"answer": "na", "justification": "the audio lacks sufficient cues to determine this feature."}
    return None

# -------------------- Load AF3 --------------------
print("[Info] Preparing AF3 model")
MODEL_DIR = os.environ.get("MODEL_DIR", "").strip()

def _is_model_dir(d: str) -> bool:
    if not os.path.isdir(d): return False
    files = set(os.listdir(d))
    has_cfg = ("config.json" in files)
    has_weight_single = any(fn.startswith("model") and fn.endswith(".safetensors") for fn in files)
    has_weight_index  = ("model.safetensors.index.json" in files)
    return has_cfg and (has_weight_single or has_weight_index)

def _find_model_dir_bfs(root: str) -> str:
    from collections import deque
    q, seen = deque([root]), set()
    while q:
        cur = q.popleft()
        if cur in seen: continue
        seen.add(cur)
        if _is_model_dir(cur): return cur
        try:
            for p in sorted(os.listdir(cur)):
                full = os.path.join(cur, p)
                if os.path.isdir(full): q.append(full)
        except Exception:
            pass
    return ""

if not MODEL_DIR or not os.path.isdir(MODEL_DIR):
    user_hub = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--audio-flamingo-3/snapshots")
    if os.path.isdir(user_hub):
        cand = sorted([os.path.join(user_hub, p) for p in os.listdir(user_hub)], key=os.path.getmtime)
        if cand: MODEL_DIR = cand[-1]
if not MODEL_DIR or not os.path.isdir(MODEL_DIR):
    print("[Info] No local MODEL_DIR found, snapshot_download weights...")
    MODEL_DIR = snapshot_download(repo_id=args.model_repo, repo_type="model", cache_dir=hf_cache)

llm_dir = _find_model_dir_bfs(MODEL_DIR)
if not llm_dir:
    raise RuntimeError(f"[ERR] AF3 LLM weights dir not found under: {MODEL_DIR}")
model_root = os.path.dirname(llm_dir)
print(f"[Info] Using AF3 model root: {model_root}")

REPO_ROOT = os.environ.get("AF3_LLAVA_REPO", "/mnt/SSD3/chonghan/repos/audio-flamingo")
if os.path.isdir(REPO_ROOT):
    sys.path.insert(0, REPO_ROOT)

from llava.entry import load as af3_load
from llava.media import Sound as AF3Sound

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device_map = "cuda" if device == "cuda" else "cpu"

model = af3_load(
    model_root,
    model_base=None,
    device_map=device_map,
    torch_dtype=dtype
)
model.eval()

generation_config = model.default_generation_config
generation_config.max_new_tokens = args.max_new_tokens
generation_config.temperature = 0.0
generation_config.top_p = 1.0
generation_config.do_sample = False
if hasattr(generation_config, "repetition_penalty"):
    generation_config.repetition_penalty = 1.0

@torch.inference_mode()
def af3_generate(audio_file_path: str, prompt: str) -> str:
    snd = AF3Sound(audio_file_path)
    res = model.generate_content([snd,prompt], generation_config=generation_config)
    if isinstance(res, dict):
        for k in ("text","content","message"):
            if isinstance(res.get(k), str): return res[k]
        try: return json.dumps(res, ensure_ascii=False)
        except Exception: return str(res)
    if isinstance(res, list) and res:
        r0 = res[0]
        if isinstance(r0, str): return r0
        try: return json.dumps(r0, ensure_ascii=False)
        except Exception: return str(r0)
    return str(res)

def extract_answers_by_af3(audio_wav: str, prompt_map: Dict[str, str],
                           max_retries: int, raw_try_collector: List[Tuple[str, str, str]]):
    out: Dict[str, Dict[str, str]] = {}
    for feature, prompt in prompt_map.items():
        ok = False
        last_raw = None

        for _ in range(max_retries):
            raw = af3_generate(audio_wav, prompt)
            last_raw = raw
            raw_try_collector.append((feature, prompt, str(raw)))

            parsed = strict_json_load(raw)
            if parsed is None:
                continue

            ans = norm_answer(parsed.get("answer", ""))
            just = str(parsed.get("justification", "") or "").strip()
            if ans in ("yes", "no", "na"):
                out[feature] = {"answer": ans, "justification": just}
                ok = True
                break

        if not ok and last_raw is not None:
            fallback = tolerant_parse_from_text(last_raw, feature_key=feature)
            if fallback and fallback.get("answer") in ("yes", "no", "na"):
                out[feature] = {
                    "answer": fallback["answer"],
                    "justification": fallback.get("justification","").strip()
                }
                ok = True

        if not ok:
            out[feature] = {"answer": "fail", "justification": ""}

        # if justification is empty, then add one sentence as default.
        ans_now  = out[feature]["answer"]
        if not out[feature]["justification"]:
            if ans_now == "na" and feature == "verbal_responsiveness":
                out[feature]["justification"] = "no clear verbal address is heard in the audio, so verbal responsiveness cannot be determined."
            elif ans_now == "na":
                out[feature]["justification"] = "the audio lacks sufficient cues to determine this feature."
            elif ans_now == "yes" and feature == "ictal_vocalization":
                out[feature]["justification"] = "the audio contains clear vocal signs such as groaning/moaning/guttural sounds."
            elif ans_now == "no" and feature == "ictal_vocalization":
                out[feature]["justification"] = "no groaning, moaning, guttural sounds, or repetitive phrases are heard."
            elif ans_now == "yes":
                out[feature]["justification"] = "audio evidence supports a 'yes' assessment."
            elif ans_now == "no":
                out[feature]["justification"] = "audio evidence supports a 'no' assessment."

    return out

# -------------------- Main --------------------
def main():
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)

    all_wavs = [f for f in os.listdir(args.dataset_dir) if f.lower().endswith(".wav")]
    all_wavs.sort()
    if args.max_groups != -1:
        all_wavs = all_wavs[:args.max_groups]

    print(f"=== ALM · Task1 (AF3 · one-pass · strict->fallback) ===")
    print(f"GPU: {args.gpu}")
    print(f"Data : {args.dataset_dir}  (files: {len(all_wavs)})")
    print(f"Out  : {args.output_dir}")
    print(f"Cache: {hf_cache}")
    print(f"Max retries per feature: {args.max_retries}")
    print("-"*60)

    for fname in tqdm(all_wavs, desc="Processing audios"):
        fpath = os.path.join(args.dataset_dir, fname)
        try:
            a, _ = load_audio_to_16k_mono(fpath, target_sr=16000)
        except Exception as e:
            print(f"[Error] Bad audio {fname}: {e}")
            save_row(CSV_PATH, [fname, "fail","Invalid audio file","fail","Invalid audio file"])
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpf:
            tmp_path = tmpf.name
            wav_tensor = torch.from_numpy(a).unsqueeze(0).to(torch.float32)
            torchaudio.save(tmp_path, wav_tensor, sample_rate=16000)

            raw_tries: List[Tuple[str, str, str]] = []
            result = extract_answers_by_af3(
                audio_wav=tmp_path,
                prompt_map={
                    "verbal_responsiveness": PROMPTS["verbal_responsiveness"],
                    "ictal_vocalization":    PROMPTS["ictal_vocalization"],
                },
                max_retries=args.max_retries,
                raw_try_collector=raw_tries
            )

            vr = result.get("verbal_responsiveness", {"answer":"fail","justification":"fail"})
            iv = result.get("ictal_vocalization",    {"answer":"fail","justification":"fail"})
            row = [fname, vr["answer"], vr["justification"], iv["answer"], iv["justification"]]
            save_row(CSV_PATH, row)

            if not args.disable_logs:
                raw_dir = os.path.join(args.output_dir, "log")
                os.makedirs(raw_dir, exist_ok=True)
                logf = os.path.join(raw_dir, f"{os.path.splitext(fname)[0]}---raw.csv")
                try:
                    with open(logf, "w", newline="", encoding="utf-8") as lf:
                        w = csv.writer(lf)
                        w.writerow(["feature","prompt","raw_text"])
                        for feat, prm, raw in raw_tries:
                            w.writerow([feat, prm, str(raw)])
                except Exception as _e:
                    print(f"[Warn] Failed to write raw log for {fname}: {_e}")

    print(f"[Done] Results saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()
