# LLMSmokeDetector

[![Paper DOI](https://img.shields.io/badge/Paper-10.1093%2Fjamiaopen%2Fooaf182-blue)](https://doi.org/10.1093/jamiaopen/ooaf182)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

<!-- After you mint a Zenodo DOI (see "Archiving a release on Zenodo" below),
     add the archive badge here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

A two-stage large language model (LLM) pipeline that extracts tobacco smoking status from free-text clinical discharge summaries and evaluates predictive performance. This is the companion code for a JAMIA Open Brief Communication comparing six lightweight open-source LLMs against the cTAKES NLP system. See the [associated publication](#associated-publication).

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quickstart](#quickstart)
- [Demo notebook](#demo-notebook)
- [Data format](#data-format)
- [CLI reference](#cli-reference)
- [Outputs](#outputs)
- [Reproducing the paper](#reproducing-the-paper)
- [Troubleshooting](#troubleshooting)
- [Associated publication](#associated-publication)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## How it works

The pipeline classifies each note into one of three labels (`Smoker`, `Never Smoker`, `Unknown`) in two stages, then scores the predictions:

1. **Stage 1 (unstructured):** each note is passed to an LLM, which returns a short free-text classification. Per-note wall-clock time is recorded.
2. **Stage 2 (structured):** the Stage 1 text is mapped to exactly one allowed label using constrained decoding (via `outlines`), with a smaller model. Per-note time is recorded.
3. **Evaluation:** multiclass and per-class metrics are computed with 1000-sample bootstrap confidence intervals, alongside confusion matrices.

## Requirements

- **Python 3.12.**
- **A CUDA-capable NVIDIA GPU.** GPU memory scales with model choice; the paper reports on-GPU sizes from roughly 2.3 GB (1B) to 47.3 GB (70B, 4-bit). A single 16 GB or larger card is sufficient for models up to gpt-oss-20b (MXFP4); the 70B model needs substantially more memory (for example, a 48 to 80 GB card or multiple GPUs). Exact per-model sizes are in the paper's supplementary tables.
- **A Hugging Face account.** Several models are gated on the Hub and require you to accept their license and authenticate:
  - Gated: the Llama-3.x family (`1B`, `3B`, `8B`, `70B`) and MedGemma (`27B`).
  - Open: gpt-oss-20b (`20B`).
  Request access on each model's Hub page, then run `huggingface-cli login`.
- **Disk space** for model weights and the Hugging Face cache.

## Installation

```bash
git clone https://github.com/ddav118/LLMSmokeDetector.git
cd LLMSmokeDetector

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` is a full environment freeze that records the exact packages used to produce the published results (CUDA 12.4, Python 3.12). Two pins are deliberate and important for the gpt-oss-20b path:

- `torch` is a CUDA 12.4 nightly build, and
- `transformers` is pinned to a specific commit,

both of which were required for gpt-oss MXFP4 support at the time of the study. If `pip install -r requirements.txt` cannot resolve the PyTorch nightly (older nightlies are periodically removed from the index), install a recent PyTorch build for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) first, then install the remaining dependencies. The pinned `transformers` commit is still required for `GptOssForCausalLM` and `Mxfp4Config`.

Then authenticate with Hugging Face so gated models can download:

```bash
huggingface-cli login
```

## Configuration

Two environment variables control where models and caches live. Both are optional and fall back to sensible defaults.

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLMSMOKE_BASE_PATH` | Root directory that may contain a `LLMs/` subfolder with local model directories. | current working directory |
| `HF_HOME` | Hugging Face cache location. | `~/.cache/huggingface` |

Model loading works two ways. If a model directory exists locally under `$LLMSMOKE_BASE_PATH/LLMs/<name>` (the names are listed in `load_model.py` under `MODEL_CONFIGS`), it loads from there. Otherwise, the loaders fall back to downloading from the Hugging Face Hub. You do not need local copies to get started; authenticate and the weights will be fetched on first use.

```bash
export LLMSMOKE_BASE_PATH=/path/to/your/workspace
export HF_HOME=/path/to/your/hf-cache
```

## Quickstart

Run Stage 1, Stage 2, then evaluate:

```bash
python run_analysis.py stage1 \
  --model 20B \
  --temperature 1.0 \
  --reasoning-effort low \
  --data-path data/test_ann_complete.csv \
  --results-dir results

python run_analysis.py stage2 \
  --in-csv results/preds/20B_temp1.0_stage1.csv \
  --structured-model-key 8B \
  --temperature 0.0 \
  --results-dir results

python run_analysis.py evaluate \
  --in-csv results/preds/20B_temp1.0_stage2.csv \
  --results-dir results
```

Format the summary table (optional):

```bash
python run_analysis.py format-table --results-dir results
```

## Demo notebook

`LLMSmokeDetector_demo.ipynb` runs the full Stage 1 plus Stage 2 flow on a single synthetic discharge note, so you can verify your environment without any protected data. Open it from the repository root (the same folder as `stage1.py` and `stage2.py`). You can switch to a smaller model (`1B` or `3B`) inside the notebook if you are GPU-constrained.

## Data format

### Stage 1 input

A CSV read into a pandas DataFrame.

- Required: a note-text column (default `full_discharge_summary`, set with `--text-col`).
- Recommended: a ground-truth label column (default `smoke_status_consensus`), needed later for `evaluate`.

Stage 1 also guarantees a row identifier (`pat_enc_csn_id`); if absent, it is created from the DataFrame index.

### Stage 1 output

Adds `stage1_raw_output` (raw model text) and `stage1_time_seconds` (per-note time). Default path: `results/preds/<MODEL>_temp<TEMP>_stage1.csv`.

### Stage 2 input and output

Stage 2 reads a Stage 1 CSV (column `stage1_raw_output`, set with `--stage1-col`) and adds `stage2_label` (one of `Smoker`, `Never Smoker`, `Unknown`) and `stage2_time_seconds`. If the Stage 1 filename matches `<MODEL>_temp<TEMP>_stage1.csv`, those values name the Stage 2 output. Default path: `results/preds/<MODEL>_temp<TEMP>_stage2.csv`.

### Evaluation input

Requires the ground-truth column (`--y-true-col`, default `smoke_status_consensus`) and the prediction column (`--y-pred-col`, default `stage2_label`) to exist in the Stage 2 CSV.

## CLI reference

All commands run through:

```bash
python run_analysis.py <command> [options]
```

### `stage1`

Run Stage 1 unstructured generation.

```bash
python run_analysis.py stage1 --model {1B,3B,8B,20B,27B,70B} [options]
```

- `--model` (required): one of `1B | 3B | 8B | 20B | 27B | 70B`
- `--temperature` (default `1.0`): sampling temperature
- `--reasoning-effort` (default `low`): forwarded into the chat template for models that support it (for example, gpt-oss)
- `--data-path` (default `data/test_ann_complete.csv`): input CSV
- `--results-dir` (default `results`): results root
- `--text-col` (default `full_discharge_summary`): note-text column
- `--out-csv` (default: derived from model and temperature): output path override
- `--window-tokens` (default `2048`): window size for the long-note fallback
- `--max-new-tokens` (default `200`): generation length cap

### `stage2`

Map Stage 1 outputs to a single structured label.

```bash
python run_analysis.py stage2 --in-csv <STAGE1_CSV> [options]
```

- `--in-csv` (required): Stage 1 CSV from the `stage1` command
- `--results-dir` (default `results`): results root
- `--stage1-col` (default `stage1_raw_output`): column with Stage 1 raw outputs
- `--structured-model-key` (default `8B`): model used for constrained decoding
- `--temperature` (default `1.0`): decoding temperature; set to `0.0` for deterministic structured decoding
- `--max-new-tokens` (default `10`): max tokens for constrained decoding
- `--out-csv` (default: derived from input filename): output path override
- `--drop-stage1` (flag): drop the Stage 1 raw output column from the Stage 2 output

Stage 2 requires the `outlines` package.

### `evaluate`

Compute full metrics from one Stage 2 CSV or a glob of them.

```bash
python run_analysis.py evaluate --in-csv <STAGE2_CSV> [options]
# or
python run_analysis.py evaluate --glob "results/preds/*_stage2.csv" [options]
```

- exactly one of `--in-csv` or `--glob` (default glob `results/preds/*_stage2.csv`)
- `--results-dir` (default `results`): results root
- `--y-true-col` (default `smoke_status_consensus`): ground-truth column
- `--y-pred-col` (default `stage2_label`): prediction column
- `--tag` (optional): override for output filenames

If `stage1_time_seconds` is present, evaluation also bootstraps Stage 1 timing into the summary row.

### `format-table`

Format the metrics summary CSV into a publication-ready table.

```bash
python run_analysis.py format-table [options]
```

- `--results-dir` (default `results`): results root
- `--in-path` (default `results/llm_metrics_summary.csv`): input summary CSV
- `--out-path` (default `results/llm_metrics_formatted_full.csv`): output CSV

## Outputs

Written under `--results-dir`:

- `llm_metrics_summary.csv`: one row per evaluated model (append mode)
- `llm_metrics_formatted_full.csv`: the same, formatted as `mean [lower, upper]`
- `clinical_metrics/per_class_metrics_<tag>.csv`: per-class PPV, sensitivity, specificity, NPV with confidence intervals
- `conf_mats/confusion_matrix_<tag>.csv`: confusion matrix (rows are true labels, columns are predicted)

## Reproducing the paper

The exact decoding settings, model versions, and evaluation choices used for the published results are described in the paper's Methods and Supplementary Methods. The underlying clinical data contain protected health information and cannot be shared publicly (see the paper's data availability statement); the `LLMSmokeDetector_demo.ipynb` notebook lets you exercise the full pipeline on a synthetic note instead.

## Troubleshooting

- **Gated-model download errors (401 or 403).** Request access on the model's Hugging Face page and run `huggingface-cli login`. This affects the Llama-3.x models and MedGemma.
- **CUDA out of memory.** Stage 1 automatically retries long notes with a smaller token window. If you still run out of memory, use a smaller `--model` or reduce `--window-tokens`.
- **A `torch.compile` / Inductor error during Stage 2** (for example, `CompiledKernel has no attribute 'launch_enter_hook'`). This stems from a PyTorch and Triton version interaction during constrained decoding. Disable compilation for the run and the pipeline proceeds without it:

  ```bash
  TORCHDYNAMO_DISABLE=1 python run_analysis.py stage2 --in-csv ... 
  ```

## Associated publication

Dávila-García DM, Schuelke MJ, Wilcox AB. Lightweight open-source large language models versus cTAKES for information extraction from discharge summaries: tobacco smoking status test case. JAMIA Open. 2026 Feb 1;9(1):ooaf182. doi:[10.1093/jamiaopen/ooaf182](https://doi.org/10.1093/jamiaopen/ooaf182)

Authors are affiliated with the Institute for Informatics, Data Science & Biostatistics, Washington University School of Medicine in St. Louis, St. Louis, MO, USA.

## Citation

If you use this software, please cite the paper above.

### BibTeX

```bibtex
@article{davila_garcia_llmsmokedetector_jamiaopen_2026,
  author  = {D{\'a}vila-Garc{\'i}a, David M and Schuelke, Matthew J and Wilcox, Adam B},
  title   = {Lightweight open-source large language models versus cTAKES for information extraction from discharge summaries: tobacco smoking status test case},
  journal = {JAMIA Open},
  volume  = {9},
  number  = {1},
  pages   = {ooaf182},
  year    = {2026},
  month   = {02},
  doi     = {10.1093/jamiaopen/ooaf182},
  issn    = {2574-2531},
  url     = {https://doi.org/10.1093/jamiaopen/ooaf182}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

The authors thank Trudy Landreth for assistance with manual annotation of discharge summaries. The authors received no external funding.
