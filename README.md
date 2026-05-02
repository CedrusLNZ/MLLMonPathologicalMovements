signal enhancement

## CNN And ViViT Benchmarks

Reusable benchmark scripts are included for feature-folder video classification:

- `finetune_vit_by_folder.py`: fine-tunes a video transformer, defaulting to
  `google/vivit-b-16x2-kinetics400`.
- `finetune_cnn_by_folder.py`: fine-tunes torchvision 3D CNN baselines such as
  `r3d_18`, `mc3_18`, and `r2plus1d_18`.
- `aggregate_patient_predictions.py` and `evaluate_patient_predictions.py`:
  aggregate segment-level predictions to patient-level scores and compute
  thresholded metrics.
- `filter_videos_by_csv.py`: renames videos from a CSV mapping and removes
  unlisted files after an explicit `--apply`.

Install the benchmark dependencies with:

```bash
python -m pip install -r requirements-benchmarks.txt
```

Generated checkpoints, prediction CSVs, and metric folders are ignored by Git.

## Prompt Optimization

`prompt_robustness.py` and `mllm_video_backend.py` contain the prompt robustness
and GEPA utilities. Generated prompt sweeps belong under `prompt_experiments/`,
which is ignored by Git.
