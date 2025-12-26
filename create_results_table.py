import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict

LOWER_SUFFIX = "_CI_Lower"
UPPER_SUFFIX = "_CI_Upper"

def find_col(df, name):
    by_lower = {c.lower(): c for c in df.columns}
    key = by_lower.get(name.lower())
    if key is None:
        raise KeyError(f"Required column '{name}' not found. Available: {list(df.columns)}")
    return key

def fmt3(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.3f}"
    except Exception:
        return ""

def fmt_triplet(mean, lo, hi):
    m = fmt3(mean)
    l = fmt3(lo) if lo is not None else ""
    h = fmt3(hi) if hi is not None else ""
    if m and l and h:
        return f"{m} [{l}, {h}]"
    elif m and (l or h):
        # degrade gracefully if only one bound is present
        return f"{m} [{l}, {h}]"
    return m

def discover_metric_bases(df, reserved=("Model", "Temperature")):
    """
    Build an ordered mapping: base_name -> dict(mean=col, lower=col_or_None, upper=col_or_None).
    Preserves first-seen column order. Ensures 'Accuracy' is listed first if present.
    """
    bases = OrderedDict()

    def ensure(base):
        if base not in bases:
            bases[base] = {"mean": None, "lower": None, "upper": None}

    # First pass: record means and CIs, preserving DF column order
    for col in df.columns:
        if col in reserved:
            continue
        if col.endswith(LOWER_SUFFIX):
            base = col[: -len(LOWER_SUFFIX)]
            ensure(base)
            bases[base]["lower"] = col
        elif col.endswith(UPPER_SUFFIX):
            base = col[: -len(UPPER_SUFFIX)]
            ensure(base)
            bases[base]["upper"] = col
        else:
            # Treat everything else as a mean metric
            base = col
            ensure(base)
            bases[base]["mean"] = col

    # Keep only bases that actually have a mean column
    bases = OrderedDict((b, v) for b, v in bases.items() if v["mean"] is not None)

    # Move Accuracy to the front if present
    if "Accuracy" in bases:
        acc = OrderedDict([("Accuracy", bases.pop("Accuracy"))])
        acc.update(bases)
        bases = acc

    return bases

def main(in_path, out_path):
    df = pd.read_csv(in_path)

    model_col = find_col(df, "Model")
    temp_col = find_col(df, "Temperature")

    bases = discover_metric_bases(df, reserved=(model_col, temp_col))
    if not bases:
        raise RuntimeError("No metric bases discovered. Check column names and CI suffixes.")

    out_rows = []
    for _, row in df.iterrows():
        rec = OrderedDict()
        rec["Model"] = row[model_col]
        rec["Temperature"] = "" if pd.isna(row[temp_col]) else fmt3(row[temp_col])

        for base, cols in bases.items():
            mean_col = cols["mean"]
            lo_col = cols["lower"]
            hi_col = cols["upper"]
            mean = row.get(mean_col, np.nan)
            lo = row.get(lo_col, np.nan) if lo_col is not None else None
            hi = row.get(hi_col, np.nan) if hi_col is not None else None
            rec[base] = fmt_triplet(mean, lo, hi)

        out_rows.append(rec)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows and {len(out_df.columns)} columns to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format all LLM metrics as 'mean [lower, upper]' with 3-dec rounding.")
    parser.add_argument("--in", dest="in_path", default="results/llm_metrics_summary.csv",
                        help="Input CSV path")
    parser.add_argument("--out", dest="out_path", default="results/llm_metrics_formatted_full.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    main(args.in_path, args.out_path)
