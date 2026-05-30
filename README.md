# LLMSmokeDetector

## Contributors

* [David M. Dávila-García, M.A.](https://orcid.org/0000-0002-9951-2270)
* [Matthew J. Schuelke, PhD](https://orcid.org/0000-0001-5755-1725)
* [Adam B. Wilcox, PhD](https://orcid.org/0000-0002-6305-735X)

### Institutional affiliation
Institute for Informatics, Data Science & Biostatistics, Washington University School of Medicine in Saint Louis, St. Louis, MO, USA

## Associated publication

This repository contains source code used in:

Dávila-García DM, Schuelke MJ, Wilcox AB. Lightweight open-source large language models versus cTAKES for information extraction from discharge summaries: tobacco smoking status test case. Jamia Open. 2026 Feb 1;9(1):ooaf182. doi:[10.1093/jamiaopen/ooaf182](https://doi.org/10.1093/jamiaopen/ooaf182)

LLMSmokeDetector is a two-stage pipeline for classifying tobacco smoking status from free-text clinical notes, then evaluating predictive performance.


Core idea:

1. Stage 1 generates an unstructured classification from each note and records time per note.
2. Stage 2 converts Stage 1 text into exactly one structured label from: `Smoker`, `Never Smoker`, `Unknown` (constrained decoding) and records time per note.
3. Evaluation computes multiclass metrics with bootstrap confidence intervals, per class clinical metrics, and confusion matrices.

## Repository layout

```text
LLMSmokeDetector/
├── README.md
├── requirements.txt
├── run_analysis.py
├── stage1.py
├── stage2.py
├── load_model.py
├── prompts.py
├── advanced_analytics.py
├── create_results_table.py
├── data/
│   └── test_ann_complete.csv
└── results/
    ├── preds/
    ├── clinical_metrics/
    ├── conf_mats/
    ├── llm_metrics_summary.csv
    └── llm_metrics_formatted_full.csv
```

Module responsibilities:

- `run_analysis.py`: CLI entrypoint. Defines subcommands and dispatches into `stage1.py`, `stage2.py`, and evaluation utilities.
- `stage1.py`: Stage 1 unstructured generation and fallback windowing for long notes or CUDA OOM.
- `stage2.py`: Stage 2 constrained decoding that maps Stage 1 output to one of the allowed labels.
- `load_model.py`: Model and tokenizer loading for both stages. Defines `OUTPUT_LABELS`. Supports `LLMSMOKE_BASE_PATH` and `HF_HOME` environment variables.
- `prompts.py`: Prompt templates for Stage 1 and Stage 2.
- `advanced_analytics.py`: Metrics, bootstrap confidence intervals, clinical metrics, and confusion matrix utilities.
- `create_results_table.py`: Formatting helper that converts the metrics summary CSV into publication ready strings (mean with confidence interval).

## Data expectations

### Stage 1 input CSV

Stage 1 reads a CSV into a pandas DataFrame.

Required column:

- Note text column (default `full_discharge_summary`, configurable via `--text-col`)

Recommended columns:

- Ground truth label column (default `smoke_status_consensus`). This is only required later for `evaluate`.

### Stage 1 output CSV

Stage 1 appends:

- `stage1_raw_output`: raw model text output for each note
- `stage1_time_seconds`: wall clock time per note (seconds)

Stage 1 also ensures a row identifier column exists:

- If `pat_enc_csn_id` exists in the input, it is preserved.
- If it does not exist, Stage 1 creates `pat_enc_csn_id` from the DataFrame index.

Default output path (unless `--out-csv` is set):

- `results/preds/<MODEL>_temp<TEMP>_stage1.csv`

### Stage 2 input CSV

Stage 2 expects a Stage 1 CSV and requires a column containing Stage 1 raw outputs:

- `stage1_raw_output` by default, configurable via `--stage1-col`

### Stage 2 output CSV

Stage 2 appends:

- `stage2_label`: one of `Smoker`, `Never Smoker`, `Unknown`
- `stage2_time_seconds`: wall clock time per note (seconds)

Stage 2 also ensures `row_id` exists (created from index if missing). It does not require Stage 1 to have `row_id`.

Default output path (unless `--out-csv` is set):

- `results/preds/<MODEL>_temp<TEMP>_stage2.csv`

Notes:

- If the Stage 1 input filename matches `<MODEL>_temp<TEMP>_stage1.csv`, Stage 2 uses those values to name the default Stage 2 output file.
- If not, Stage 2 defaults to `stage1_temp1.0_stage2.csv` naming for the inferred model and temperature.

### Evaluation inputs

Evaluation requires:

- `--y-true-col` (default `smoke_status_consensus`)
- `--y-pred-col` (default `stage2_label`)

Evaluation expects that both columns exist in the Stage 2 output CSV.

## Quickstart

Run Stage 1, Stage 2, then evaluate:

```bash
python run_analysis.py stage1   --model 20B   --temperature 1.0   --reasoning-effort low   --data-path data/test_ann_complete.csv   --results-dir results

python run_analysis.py stage2   --in-csv results/preds/20B_temp1.0_stage1.csv   --structured-model-key 8B   --results-dir results

python run_analysis.py evaluate   --in-csv results/preds/20B_temp1.0_stage2.csv   --results-dir results
```

Format the summary table (optional):

```bash
python run_analysis.py format-table --results-dir results
```

## CLI reference (run_analysis.py)

All commands are executed through:

```bash
python run_analysis.py <command> [options]
```

### Command: stage1

Run Stage 1 unstructured generation and save outputs to CSV.

Syntax:

```bash
python run_analysis.py stage1 --model {1B,3B,8B,20B,27B,70B} [options]
```

Arguments:

- `--model` (required): `1B | 3B | 8B | 20B | 27B | 70B`
- `--temperature` (default `1.0`): sampling temperature
- `--reasoning-effort` (default `low`): forwarded into the chat template for models that support it
- `--data-path` (default `data/test_ann_complete.csv`): input CSV path
- `--results-dir` (default `results`): results root directory
- `--text-col` (default `full_discharge_summary`): column containing the note text
- `--out-csv` (default: derived from model and temperature): output CSV path override
- `--window-tokens` (default `2048`): window size (tokens) for the fallback chunking strategy
- `--max-new-tokens` (default `200`): generation length cap for Stage 1

Outputs:

- CSV at `--out-csv` or `results/preds/<MODEL>_temp<TEMP>_stage1.csv`
- Adds `stage1_raw_output` and `stage1_time_seconds`

### Command: stage2

Run Stage 2 structured sanitization from a Stage 1 CSV.

Syntax:

```bash
python run_analysis.py stage2 --in-csv <STAGE1_CSV> [options]
```

Arguments:

- `--in-csv` (required): Stage 1 CSV produced by the `stage1` command
- `--results-dir` (default `results`): results root directory
- `--stage1-col` (default `stage1_raw_output`): column containing Stage 1 raw outputs
- `--structured-model-key` (default `8B`): model key used for constrained decoding
- `--temperature` (default `0.0`): decoding temperature (commonly `0.0`)
- `--max-new-tokens` (default `10`): max tokens for constrained decoding
- `--out-csv` (default: derived from input filename): output CSV path override
- `--drop-stage1` (flag): if set, drop the Stage 1 raw output column in the Stage 2 output

Outputs:

- CSV at `--out-csv` or `results/preds/<MODEL>_temp<TEMP>_stage2.csv`
- Adds `stage2_label` and `stage2_time_seconds`
- Preserves `stage1_time_seconds` if it was present in the Stage 1 input CSV

Dependency note:

- Stage 2 requires the `outlines` package.

### Command: evaluate

Compute full performance metrics from one Stage 2 CSV or from a glob of Stage 2 CSVs.

Syntax (single file):

```bash
python run_analysis.py evaluate --in-csv <STAGE2_CSV> [options]
```

Syntax (glob):

```bash
python run_analysis.py evaluate --glob "results/preds/*_stage2.csv" [options]
```

Arguments:

- Exactly one of:
  - `--in-csv`: Stage 2 CSV to evaluate
  - `--glob` (default `results/preds/*_stage2.csv`): glob pattern for Stage 2 CSVs
- `--results-dir` (default `results`): results root directory
- `--y-true-col` (default `smoke_status_consensus`): ground truth column name
- `--y-pred-col` (default `stage2_label`): prediction column name
- `--tag` (optional): override for output filenames (per class metrics and confusion matrix names)

Outputs (written under `--results-dir`):

- `llm_metrics_summary.csv` (append mode)
- `clinical_metrics/per_class_metrics_<tag>.csv`
- `conf_mats/confusion_matrix_<tag>.csv`

Notes:

- If `stage1_time_seconds` exists in the Stage 2 CSV, evaluation also bootstraps Stage 1 timing and writes `Stage1_TimeSeconds_*` columns into the summary row.
- Evaluation may attempt to generate `llm_metrics_formatted_full.csv` automatically if `create_results_table.py` is importable. The `format-table` command provides an explicit way to do this.

### Command: format-table

Format the metrics summary CSV into a publication ready table.

Syntax:

```bash
python run_analysis.py format-table [options]
```

Arguments:

- `--results-dir` (default `results`): results root directory
- `--in-path` (default `results/llm_metrics_summary.csv`): input summary CSV path
- `--out-path` (default `results/llm_metrics_formatted_full.csv`): output formatted CSV path

Outputs:

- `llm_metrics_formatted_full.csv` at the output path

## Help

```bash
python run_analysis.py -h
python run_analysis.py stage1 -h
python run_analysis.py stage2 -h
python run_analysis.py evaluate -h
python run_analysis.py format-table -h
```

## Citation

When using this resource, please cite:

Dávila-García DM, Schuelke MJ, Wilcox AB. Lightweight open-source large language models versus cTAKES for information extraction from discharge summaries: tobacco smoking status test case. Jamia Open. 2026 Feb 1;9(1):ooaf182. doi:[10.1093/jamiaopen/ooaf182](https://doi.org/10.1093/jamiaopen/ooaf182)

### BibTeX

```bibtex
@article{10.1093/jamiaopen/ooaf182,
    author = {Dávila-García, David M and Schuelke, Matthew J and Wilcox, Adam B},
    title = {Lightweight open-source large language models versus cTAKES for information extraction from discharge summaries: tobacco smoking status test case},
    journal = {JAMIA Open},
    volume = {9},
    number = {1},
    pages = {ooaf182},
    year = {2026},
    month = {02},
    abstract = {To compare lightweight open-source large language models (LLMs) with cTAKES, a state-of-the-art natural language processing (NLP) system, in an information extraction task from hospitalization discharge summaries.Two readers annotated 250 randomly sampled adult discharge summaries (BJC HealthCare, 2018-2023) for tobacco smoking status as “Smoker,” “Never smoker,” “Unknown.” Six LLMs (Llama-3 [1B-70B], gpt-oss-20B, MedGemma-27B) and cTAKES extracted smoking status from summaries. Performance was benchmarked against consensus annotations using weighted F1-score, macro F1-score, and per-class F1-scores and a noninferiority test.Inter-reader agreement was excellent (κ = 0.91). LLM size (2.3-47.3 GB) and inference time (2.5-14.5 s/note) varied. gpt-oss-20B achieved non-inferior performance vs cTAKES (F1 = 0.99 vs 0.97; P \&lt; .021).The high accuracy and efficiency of gpt-oss-20B support its potential as a practical, open-source alternative to traditional NLP for clinical information extraction.Lightweight LLMs can be applied for use across diverse clinical information extraction tasks without the need for task-specific fine-tuning.Electronic health records contain large amounts of patient-related information, such as tobacco smoking status and/or other important details regarding patient behaviors. This information is typically written in an unstructured and/or free-text form, making it difficult to extract for downstream use to improve patient care, reporting, and research. Extracting such information accurately and efficiently allows clinicians and researchers to improve patient care and will ultimately benefit population health. In this study, we compared cTAKES, a state-of-the-art natural language processing (NLP) system, against 6 lightweight open-source large language models (LLMs) to extract tobacco smoking status from 250 hospital discharge summaries. After showing excellent agreement for extracting smoking status annotation from hospital discharge records by 2 readers, we showed that a 2-stage LLM inference approach accurately generated structured outputs from unstructured free-text. Finally, we found that the top-performing LLM, gpt-oss-20B, exhibited strong performance similar to cTAKES on the clinical information extraction task. These findings suggest that open-source LLMs accurately perform clinical information extraction tasks in a secure, low-cost, and practical approach. Extending this approach to perform other information extraction tasks will facilitate healthcare systems to unlock the use of unstructured data to enhance clinical care, reporting and research.},
    issn = {2574-2531},
    doi = {10.1093/jamiaopen/ooaf182},
    url = {https://doi.org/10.1093/jamiaopen/ooaf182},
    eprint = {https://academic.oup.com/jamiaopen/article-pdf/9/1/ooaf182/66416210/ooaf182.pdf},
}
```
