"""
Stage 1 (unstructured) inference for LLMSmokeDetector.

This module:
- Generates an initial unstructured smoking status classification from each clinical note
- Records wall-clock time per note
- Provides a windowed fallback for long notes or CUDA OOM

Prompt templates are defined in prompts.py
Model loading is defined in load_model.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import gc
import time
from tqdm import tqdm

import pandas as pd
import torch

from load_model import load_stage1_model_and_tokenizer
from prompts import prompt_setup


@dataclass(frozen=True)
class Stage1Columns:
    """Standard column names produced by Stage 1."""
    raw_output: str = "stage1_raw_output"
    time_seconds: str = "stage1_time_seconds"
    row_id: str = "pat_enc_csn_id"


def _decode_new_tokens(tokenizer, outputs, prompt_input_ids_len: int) -> str:
    """Decode only newly generated tokens and perform light cleanup for OSS-style outputs."""
    text = tokenizer.decode(outputs[0][prompt_input_ids_len:], skip_special_tokens=False).strip()
    # Some OSS chat templates may include channel delimiters
    text = text.split("<|channel|>final<|message|>")[-1].strip()
    return text.replace("<|return|>", "").replace("*", "").strip()


def generate_unstructured(
    note_text: str,
    model,
    tokenizer,
    temperature: float,
    reasoning_effort: str | None = "low",
    max_new_tokens: int = 200,
) -> Tuple[str, float]:
    """
    Stage 1 generation for a single note.

    Returns (raw_output, time_seconds).
    """
    device = next(model.parameters()).device if hasattr(model, "parameters") else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    messages = prompt_setup(note_text=note_text, reasoning_effort=reasoning_effort)

    template_kwargs = dict(
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort

    start = time.perf_counter()
    chat = tokenizer.apply_chat_template(messages, **template_kwargs)
    # Some tokenizers return a BatchEncoding that supports .to(device)
    chat = chat.to(device) if hasattr(chat, "to") else chat
    if isinstance(chat, dict):
        chat.pop("token_type_ids", None)

    with torch.inference_mode():
        outputs = model.generate(**chat, max_new_tokens=max_new_tokens, temperature=temperature)

    elapsed = time.perf_counter() - start
    prompt_len = chat["input_ids"].shape[-1]
    return _decode_new_tokens(tokenizer, outputs, prompt_len), float(elapsed)


def _looks_like_label(text: str) -> bool:
    """Heuristic used by the windowed fallback to short-circuit."""
    t = text.strip().lower()
    return t in {"smoker", "never smoker", "unknown"}


def classify_with_windows(
    note_text: str,
    model,
    tokenizer,
    temperature: float,
    window_tokens: int = 2048,
    reasoning_effort: Optional[str] = None,
    max_new_tokens: int = 200,
) -> Tuple[str, float]:
    """
    Stage 1 fallback for long notes or CUDA OOM.

    Splits the note into token windows (from end to start), generating on each chunk.
    Returns (raw_output, total_time_seconds).
    """
    ids = tokenizer(note_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    i = ids.size(0)

    best_text = "Unknown"
    total_time = 0.0

    while i > 0:
        j = max(0, i - window_tokens)
        chunk = tokenizer.decode(ids[j:i], skip_special_tokens=False)

        try:
            text, dt = generate_unstructured(
                note_text=chunk,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                max_new_tokens=max_new_tokens,
            )
            total_time += dt
        except torch.cuda.OutOfMemoryError:
            # Reduce window size and retry
            torch.cuda.empty_cache()
            gc.collect()
            if window_tokens > 10:
                return classify_with_windows(
                    note_text=note_text,
                    model=model,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    window_tokens=max(10, window_tokens // 2),
                    reasoning_effort=reasoning_effort,
                    max_new_tokens=max_new_tokens,
                )
            return "Unknown", total_time

        # If the model returns an exact label, short-circuit
        if _looks_like_label(text) and text.strip() in {"Smoker", "Never Smoker"}:
            return text.strip(), total_time

        best_text = text
        i = j

    return best_text, total_time


def run_stage1(
    df: pd.DataFrame,
    model_key: str,
    temperature: float = 1.0,
    reasoning_effort: str | None = "low",
    text_col: str = "full_discharge_summary",
    out_csv: Optional[str] = None,
    window_tokens: int = 2048,
    max_new_tokens: int = 200,
) -> pd.DataFrame:
    """
    Run Stage 1 inference over an in-memory dataframe.

    Appends:
    - stage1_raw_output
    - stage1_time_seconds

    Optionally writes the resulting dataframe to out_csv.
    """
    if text_col not in df.columns:
        raise ValueError(f"Missing required column: {text_col}")

    model, tokenizer = load_stage1_model_and_tokenizer(model_key)

    cols = Stage1Columns()
    df_out = df.copy()

    # Prefer an explicit row_id column if present, otherwise use index
    if cols.row_id not in df_out.columns:
        df_out[cols.row_id] = df_out.index.astype(int)

    raw_outputs: list[str] = []
    times: list[float] = []

    for _, row in tqdm(df_out.iterrows(), total=len(df_out)):
        note_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        try:
            out, dt = generate_unstructured(
                note_text=note_text,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                max_new_tokens=max_new_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            out, dt = classify_with_windows(
                note_text=note_text,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                window_tokens=window_tokens,
                reasoning_effort=reasoning_effort,
                max_new_tokens=max_new_tokens,
            )

        raw_outputs.append(out)
        times.append(float(dt))

    df_out[cols.raw_output] = raw_outputs
    df_out[cols.time_seconds] = times

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)

    return df_out