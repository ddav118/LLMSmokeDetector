"""
Stage 2 (structured) sanitization for LLMSmokeDetector.

This module:
- Converts Stage 1 unstructured outputs into a single structured label using constrained decoding
- Records wall-clock time per note for Stage 2 sanitization
- Operates on an in-memory pandas DataFrame and can optionally save outputs to CSV

Prompt templates are defined in prompts.py
Model loading is defined in load_model.py
Helper utilities should live in utils.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Literal

import time

import pandas as pd
import torch

from load_model import load_stage2_structured_generator, OUTPUT_LABELS
from prompts import prompt_sanitize

try:
    from outlines import Generator
except Exception as e:  # pragma: no cover
    Generator = None  # type: ignore[assignment]


STAGE2_LABEL_COL = "stage2_label"
STAGE2_TIME_COL = "stage2_time_seconds"
ROW_ID_COL = "row_id"


def generate_structured(
    stage1_raw_output: str,
    generator,
    tokenizer,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> Tuple[str, float]:
    """
    Stage 2 constrained decoding for a single Stage 1 output.

    Returns (label, time_seconds).
    """
    if Generator is None:
        raise ImportError(
            "Outlines is not available. Install outlines to use Stage 2 constrained decoding."
        )

    messages = prompt_sanitize(raw_output=stage1_raw_output)

    start = time.perf_counter()
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        return_tensors='pt',
        add_bos=True,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    with torch.inference_mode():
        label = generator(prompt_text, temperature=temperature, max_new_tokens=max_new_tokens)
    
    elapsed = time.perf_counter() - start
    return label, float(elapsed)


def run_stage2(
    df: pd.DataFrame,
    stage1_col: str = "stage1_raw_output",
    out_csv: Optional[str] = None,
    keep_stage1: bool = True,
    structured_model_key: str = "8B",
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> pd.DataFrame:
    """
    Run Stage 2 sanitization over an in-memory dataframe.

    Appends:
    - stage2_label
    - stage2_time_seconds

    Optionally writes the resulting dataframe to out_csv.
    """
    if stage1_col not in df.columns:
        raise ValueError(f"Missing required column for Stage 2 input: {stage1_col}")

    outlines_model, tokenizer = load_stage2_structured_generator(model_key=structured_model_key)

    if Generator is None:
        raise ImportError(
            "Outlines is not available. Install outlines to use Stage 2 constrained decoding."
        )

    # Constrain output to the allowed labels
    label_generator = Generator(outlines_model, Literal[OUTPUT_LABELS[0], OUTPUT_LABELS[1], OUTPUT_LABELS[2]])

    df_out = df.copy()

    # Prefer an explicit row_id column if present, otherwise use index
    if ROW_ID_COL not in df_out.columns:
        df_out[ROW_ID_COL] = df_out.index.astype(int)

    labels: list[str] = []
    times: list[float] = []

    for _, row in df_out.iterrows():
        raw = str(row[stage1_col]) if pd.notna(row[stage1_col]) else ""
        label, dt = generate_structured(
            stage1_raw_output=raw,
            generator=label_generator,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        labels.append(label)
        times.append(float(dt))

    df_out[STAGE2_LABEL_COL] = labels
    df_out[STAGE2_TIME_COL] = times

    if not keep_stage1 and stage1_col in df_out.columns:
        df_out = df_out.drop(columns=[stage1_col])

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)

    return df_out


__all__ = [
    "generate_structured",
    "run_stage2",
    "STAGE2_LABEL_COL",
    "STAGE2_TIME_COL",
    "ROW_ID_COL",
]
