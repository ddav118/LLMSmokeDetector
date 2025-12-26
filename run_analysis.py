"""
CLI runner for LLMSmokeDetector.

This script is intentionally minimal: it wires CLI subcommands to the project
modules that implement the actual logic.

Subcommands
-----------
- stage1: run Stage 1 unstructured generation over an input CSV (loaded to a DataFrame)
- stage2: run Stage 2 structured sanitization over a Stage 1 output CSV
- evaluate: compute full performance metrics from a Stage 2 output CSV (or glob)
- format-table: format the metrics summary CSV into a publication-ready table

Notes
-----
- Stage 1 records time per note in the output (stage1_time_seconds).
- Full predictive performance metrics are computed only after Stage 2.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from stage1 import run_stage1
from stage2 import run_stage2
from advanced_analytics import (
    compute_comprehensive_metrics,
    compute_per_class_clinical_metrics,
    save_confusion_matrix,
    bootstrap_efficiency_metrics,
)
from load_model import OUTPUT_LABELS

# Optional formatting (kept compatible with existing create_results_table.py)
try:
    from create_results_table import main as _format_results_table
except Exception:  # pragma: no cover
    _format_results_table = None


DEFAULT_DATA_PATH = "data/test_ann_complete.csv"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_TEXT_COL = "full_discharge_summary"
DEFAULT_YTRUE_COL = "smoke_status_consensus"
DEFAULT_STAGE1_COL = "stage1_raw_output"
DEFAULT_STAGE2_COL = "stage2_label"

SUMMARY_CSV = "llm_metrics_summary.csv"
FORMATTED_CSV = "llm_metrics_formatted_full.csv"


def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(p)


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _default_stage1_out(results_dir: str, model_key: str, temperature: float) -> str:
    return str(Path(results_dir) / "preds" / f"{model_key}_temp{temperature}_stage1.csv")


def _default_stage2_out(results_dir: str, model_key: str, temperature: float) -> str:
    return str(Path(results_dir) / "preds" / f"{model_key}_temp{temperature}_stage2.csv")


def _parse_model_temp_from_filename(path: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Extract model key and temperature from filenames like:
    - 20B_temp1.0_stage1.csv
    - 20B_temp1.0_stage2.csv
    """
    name = Path(path).name
    m = re.search(r"(?P<model>[A-Za-z0-9]+)_temp(?P<temp>[0-9.]+)_stage[12]\.csv", name)
    if not m:
        return None, None
    model = m.group("model")
    try:
        temp = float(m.group("temp"))
    except Exception:
        temp = None
    return model, temp


def _append_or_write_summary(summary_path: str, row: dict) -> None:
    out_path = Path(summary_path)
    if out_path.exists():
        df = pd.read_csv(out_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out_path, index=False)


def cmd_stage1(args: argparse.Namespace) -> int:
    df = _read_csv(args.data_path)

    _ensure_dir(Path(args.results_dir) / "preds")
    out_csv = args.out_csv or _default_stage1_out(args.results_dir, args.model, args.temperature)

    _ = run_stage1(
        df=df,
        model_key=args.model,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        text_col=args.text_col,
        out_csv=out_csv,
        window_tokens=args.window_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    print(out_csv)
    return 0


def cmd_stage2(args: argparse.Namespace) -> int:
    df = _read_csv(args.in_csv)

    # infer model/temp for default output naming
    model_key, temp = _parse_model_temp_from_filename(args.in_csv)
    if model_key is None:
        model_key = "stage1"
    if temp is None:
        temp = 1.0

    _ensure_dir(Path(args.results_dir) / "preds")
    out_csv = args.out_csv or _default_stage2_out(args.results_dir, model_key, temp)

    _ = run_stage2(
        df=df,
        stage1_col=args.stage1_col,
        out_csv=out_csv,
        keep_stage1=not args.drop_stage1,
        structured_model_key=args.structured_model_key,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(out_csv)
    return 0


def _evaluate_one_file(
    stage2_csv: str,
    results_dir: str,
    y_true_col: str,
    y_pred_col: str,
    tag: Optional[str] = None,
    summary_out: Optional[str] = None,
    formatted_out: Optional[str] = None,
) -> None:
    df = _read_csv(stage2_csv)

    if y_true_col not in df.columns:
        raise ValueError(f"Missing y_true column '{y_true_col}' in: {stage2_csv}")
    if y_pred_col not in df.columns:
        raise ValueError(f"Missing y_pred column '{y_pred_col}' in: {stage2_csv}")

    y_true = df[y_true_col].astype(str).to_numpy()
    y_pred = df[y_pred_col].astype(str).to_numpy()

    model_key, temp = _parse_model_temp_from_filename(stage2_csv)
    row_tag = tag or (f"{model_key}_temp{temp}" if model_key is not None and temp is not None else Path(stage2_csv).stem)

    # Full predictive metrics (Stage 2)
    metrics = compute_comprehensive_metrics(y_true, y_pred, OUTPUT_LABELS)
    metrics_row = {"Model": model_key or row_tag, "Temperature": temp if temp is not None else np.nan}
    metrics_row.update(metrics)

    # Stage 1 timing summary (if present in the Stage 2 file)
    if "stage1_time_seconds" in df.columns:
        vals = pd.to_numeric(df["stage1_time_seconds"], errors="coerce").dropna().values
        mean_t, (lo_t, hi_t) = bootstrap_efficiency_metrics(vals)
        metrics_row["Stage1_TimeSeconds_Mean"] = mean_t
        metrics_row["Stage1_TimeSeconds_CI_Lower"] = lo_t
        metrics_row["Stage1_TimeSeconds_CI_Upper"] = hi_t

    # Write summary CSV (append)
    _ensure_dir(results_dir)
    summary_path = str(Path(results_dir) / SUMMARY_CSV) if summary_out is None else summary_out
    _append_or_write_summary(summary_path, metrics_row)

    # Per-class clinical metrics + confusion matrix
    _ensure_dir(Path(results_dir) / "clinical_metrics")
    _ensure_dir(Path(results_dir) / "conf_mats")

    per_class_df = compute_per_class_clinical_metrics(y_true, y_pred, OUTPUT_LABELS)
    per_class_df.to_csv(Path(results_dir) / "clinical_metrics" / f"per_class_metrics_{row_tag}.csv", index=False)

    save_confusion_matrix(y_true, y_pred, OUTPUT_LABELS, row_tag, results_dir=str(Path(results_dir) / "conf_mats"))

    # Optional formatting of the summary table
    if _format_results_table is not None:
        try:
            formatted_path = str(Path(results_dir) / FORMATTED_CSV) if formatted_out is None else formatted_out
            _format_results_table(in_path=summary_path, out_path=formatted_path)
        except Exception:
            pass


def cmd_evaluate(args: argparse.Namespace) -> int:
    paths: List[str] = []
    if args.in_csv is not None:
        paths = [args.in_csv]
    else:
        paths = sorted(glob.glob(args.glob))
    if len(paths) == 0:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    for p in paths:
        _evaluate_one_file(
            stage2_csv=p,
            results_dir=args.results_dir,
            y_true_col=args.y_true_col,
            y_pred_col=args.y_pred_col,
            tag=args.tag,
        )
    print(str(Path(args.results_dir) / SUMMARY_CSV))
    return 0


def cmd_format_table(args: argparse.Namespace) -> int:
    if _format_results_table is None:
        raise ImportError("create_results_table.py is not available to format tables.")
    in_path = args.in_path or str(Path(args.results_dir) / SUMMARY_CSV)
    out_path = args.out_path or str(Path(args.results_dir) / FORMATTED_CSV)

    _format_results_table(in_path=in_path, out_path=out_path)
    print(out_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_analysis.py", description="LLMSmokeDetector runner (Stage 1, Stage 2, evaluation).")
    sub = p.add_subparsers(dest="command", required=True)

    # stage1
    p1 = sub.add_parser("stage1", help="Run Stage 1 unstructured generation and save outputs to CSV.")
    p1.add_argument("--model", required=True, choices=["1B", "3B", "8B", "20B", "27B", "70B"], help="Stage 1 model key.")
    p1.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for Stage 1 generation.")
    p1.add_argument("--reasoning-effort", type=str, default="low", help="Reasoning effort (used by some models).")
    p1.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="Input CSV path.")
    p1.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Root results directory.")
    p1.add_argument("--text-col", type=str, default=DEFAULT_TEXT_COL, help="Column containing note text.")
    p1.add_argument("--out-csv", type=str, default=None, help="Optional output CSV path for Stage 1.")
    p1.add_argument("--window-tokens", type=int, default=2048, help="Token window size for fallback chunking.")
    p1.add_argument("--max-new-tokens", type=int, default=200, help="Max new tokens for Stage 1 generation.")
    p1.set_defaults(func=cmd_stage1)

    # stage2
    p2 = sub.add_parser("stage2", help="Run Stage 2 structured sanitization from a Stage 1 CSV.")
    p2.add_argument("--in-csv", required=True, type=str, help="Stage 1 CSV produced by stage1 command.")
    p2.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Root results directory.")
    p2.add_argument("--stage1-col", type=str, default=DEFAULT_STAGE1_COL, help="Column containing Stage 1 raw outputs.")
    p2.add_argument("--structured-model-key", type=str, default="8B", help="Model key for Stage 2 constrained decoding.")
    p2.add_argument("--temperature", type=float, default=1.0, help="Temperature for constrained decoding (usually 0).")
    p2.add_argument("--max-new-tokens", type=int, default=10, help="Max new tokens for Stage 2 decoding.")
    p2.add_argument("--out-csv", type=str, default=None, help="Optional output CSV path for Stage 2.")
    p2.add_argument("--drop-stage1", action="store_true", help="Drop Stage 1 raw output column in the Stage 2 output.")
    p2.set_defaults(func=cmd_stage2)

    # evaluate
    p3 = sub.add_parser("evaluate", help="Compute full performance metrics from Stage 2 outputs.")
    g = p3.add_mutually_exclusive_group(required=True)
    g.add_argument("--in-csv", type=str, default=None, help="Stage 2 CSV to evaluate.")
    g.add_argument("--glob", type=str, default="results/preds/*_stage2.csv", help="Glob of Stage 2 CSVs to evaluate.")
    p3.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Root results directory.")
    p3.add_argument("--y-true-col", type=str, default=DEFAULT_YTRUE_COL, help="Ground truth column name.")
    p3.add_argument("--y-pred-col", type=str, default=DEFAULT_STAGE2_COL, help="Prediction column name.")
    p3.add_argument("--tag", type=str, default=None, help="Optional tag override for output filenames.")
    p3.set_defaults(func=cmd_evaluate)

    # format table
    p4 = sub.add_parser("format-table", help="Format metrics summary CSV into a publication-ready table.")
    p4.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Root results directory.")
    p4.add_argument("--in-path", type=str, default=None, help="Input summary CSV (defaults to results/llm_metrics_summary.csv).")
    p4.add_argument("--out-path", type=str, default=None, help="Output formatted CSV (defaults to results/llm_metrics_formatted_full.csv).")
    p4.set_defaults(func=cmd_format_table)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
