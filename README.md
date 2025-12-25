# LLMSmokeDetector

LLMSmokeDetector is a lightweight, reproducible pipeline for large language model based classification and evaluation of smoking status from clinical text.

---

## Project structure

```text
LLMSmokeDetector/
├── README.md
├── requirements.txt
├── run_analysis.py
├── load_model.py
├── pipeline.py
├── prompts.py
├── utils.py
├── advanced_analytics.py
├── data/
│   └── test_ann_complete.csv
└── results/
    ├── preds/
    ├── clinical_metrics/
    ├── conf_mats/
    ├── llm_metrics_summary.csv
    └── llm_metrics_formatted_full.csv
```

---

## CLI usage

All commands are executed through `run_analysis.py`.

```bash
python run_analysis.py <command> [options]
```

### Workflow overview (two stage, sequential, separate commands)

1. **Stage 1 (infer)**  
   LLM generates an initial unstructured smoking status classification from each note.

2. **Stage 2 (sanitize)**  
   Secondary constrained decoding model converts Stage 1 raw output into a structured label.

You run these separately so Stage 2 can be rerun without repeating Stage 1.

---

## Command: infer (Stage 1)

Runs Stage 1 inference for one model and writes raw outputs plus time per note.

```bash
python run_analysis.py infer \
  --model 20B \
  --temperature 1.0 \
  --reasoning-effort low \
  --data-path data/test_ann_complete.csv \
  --results-dir results
```

### Arguments

- `--model` (required): `1B | 3B | 8B | 20B | 27B | 70B`
- `--temperature` (default: `1.0`): sampling temperature for generation
- `--reasoning-effort` (default: `low`): only applied for 20B, ignored otherwise
- `--data-path` (default: `data/test_ann_complete.csv`): input dataset
- `--results-dir` (default: `results`): output root directory

### Outputs

- `results/preds/<model>_temp<temperature>_preds_raw.csv`

### Recommended columns in `*_preds_raw.csv`

- `row_id`: integer row index from the input dataframe
- `y_true`: ground truth label from `smoke_status_consensus`
- `raw_output`: raw generation output from Stage 1 model
- `time_seconds`: elapsed time per note (float)

---

## Command: sanitize (Stage 2)

Runs Stage 2 constrained decoding to turn Stage 1 raw outputs into structured labels.

```bash
python run_analysis.py sanitize \
  --raw-preds-glob "results/preds/*_preds_raw.csv" \
  --out-path results/preds/sanitized_outputs.csv \
  --data-path data/test_ann_complete.csv
```

### Arguments

- `--raw-preds-glob` (default: `results/preds/*_preds_raw.csv`)
- `--out-path` (default: `results/preds/sanitized_outputs.csv`)
- `--data-path` (default: `data/test_ann_complete.csv`)

### Output

- `results/preds/sanitized_outputs.csv`

### Recommended columns in `sanitized_outputs.csv`

- `smoke_status_consensus`
- one column per Stage 1 run, for example: `20B_temp1.0_sanitized`

---

## Command: aggregate (Stage 1 efficiency summaries)

Computes efficiency summaries for Stage 1 outputs that exist in `results/preds/`.

This step is limited to time-based profiling and does not compute predictive
performance metrics.

```bash
python run_analysis.py aggregate \
  --data-path data/test_ann_complete.csv \
  --results-dir results
```

### Outputs

- `results/llm_metrics_summary.csv`
- `results/llm_metrics_formatted_full.csv`
- `results/clinical_metrics/per_class_metrics_<tag>.csv`
- `results/conf_mats/confusion_matrix_<tag>.csv`

Aggregation includes mean and bootstrap CI for `time_seconds` using
`advanced_analytics.bootstrap_efficiency_metrics`.

---

## Command: sanitize-aggregate (Stage 2 metrics)

Computes metrics using structured labels in `sanitized_outputs.csv`.

```bash
python run_analysis.py sanitize-aggregate \
  --sanitized-path results/preds/sanitized_outputs.csv \
  --results-dir results
```

### Outputs

Same set of outputs as `aggregate`, derived from Stage 2 labels.

---

## Help

```bash
python run_analysis.py -h
python run_analysis.py infer -h
python run_analysis.py sanitize -h
python run_analysis.py aggregate -h
python run_analysis.py sanitize-aggregate -h
```

---

## Code reference (functions, args, one line descriptions)

### run_analysis.py (CLI runner)

- `build_parser() -> argparse.ArgumentParser`  
  Builds the CLI parser with subcommands and arguments.

- `main(argv: list[str] | None = None) -> int`  
  Parses CLI args and dispatches to the appropriate pipeline function.

Dispatch mapping:
- `infer` -> `pipeline.run_stage1_inference(...)`
- `sanitize` -> `pipeline.run_stage2_sanitization(...)`
- `aggregate` -> `pipeline.aggregate_stage1_results(...)`
- `sanitize-aggregate` -> `pipeline.aggregate_stage2_results(...)`

---

## License

MIT License. See the `LICENSE` file for details.


