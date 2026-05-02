signal enhancement

## Prompt Robustness And GEPA Optimization

`prompt_robustness.py` builds fold-aware yes/no prompt manifests, runs prompt
variants through a video-MLLM backend, scores prompt sensitivity, and can run
GEPA prompt optimization. `mllm_video_backend.py` is a persistent JSONL backend
wrapper so large Qwen or InternVL models are loaded once across many prompt
evaluations.

Generated outputs belong under `prompt_experiments/`, which is intentionally
ignored by Git. Commit only reusable code and hand-written documentation, not
prompt sweeps, model predictions, caches, checkpoints, or local credentials.

Build a manifest:

```bash
python prompt_robustness.py build-manifest \
  --csv evaluation/dataset/90_FeatureAnnotation.csv \
  --folds-json /path/to/folds.json \
  --data-root /path/to/videos \
  --output-dir prompt_experiments
```

Smoke-test the runner without loading an MLLM:

```bash
python prompt_robustness.py run \
  --examples-jsonl prompt_experiments/examples.jsonl \
  --prompts-jsonl prompt_experiments/prompts.jsonl \
  --output-jsonl /tmp/prompt_predictions_smoke.jsonl \
  --features arm_flexion \
  --prompt-ids minimal \
  --splits dev \
  --limit 2 \
  --persistent-backend-command "python mllm_video_backend.py --backend dummy" \
  --overwrite
```

Run GEPA optimization after installing the optional dependencies:

```bash
python -m pip install -r requirements-prompt-optimization.txt

python prompt_robustness.py optimize-gepa \
  --examples-jsonl prompt_experiments/examples.jsonl \
  --output-dir prompt_experiments/gepa \
  --persistent-backend-command "python mllm_video_backend.py --backend qwen --gpu 0,1 --cache-dir /path/to/cache" \
  --reflection-lm bedrock/minimax.minimax-m2.5 \
  --max-metric-calls 12 \
  --max-train-examples 8
```

For each feature and fold, GEPA optimizes on the training split only, then
evaluates the optimized prompt on dev and held-out test examples.
