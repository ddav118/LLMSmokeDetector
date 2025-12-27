"""
Model and tokenizer loading utilities for LLMSmokeDetector.

This module is the single source of truth for:
- Stage 1: unstructured LLM inference model + tokenizer
- Stage 2: constrained-decoding generator + tokenizer (via Outlines)

This module also prints GPU memory information after each successful load
using `print_gpu_info()`. It does not compute or record timing or
performance metrics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GptOssForCausalLM,
    LlamaForCausalLM,
    Mxfp4Config,
    PreTrainedTokenizerFast,
)


# Optional dependency used for constrained decoding in Stage 2.
try:
    from outlines import from_transformers
except Exception:  # pragma: no cover
    from_transformers = None  # type: ignore[assignment]


# -------------------------
# Configuration
# -------------------------

# Default locations match the current project scripts.
DEFAULT_BASE_PATH = "/storage2/fs1/a.wilcox/Active"
DEFAULT_HF_HOME = "/storage2/fs1/a.wilcox/Active/huggingface_cache"

# Allow override without code edits.
BASE_PATH = os.environ.get("LLMSMOKE_BASE_PATH", DEFAULT_BASE_PATH)
HF_HOME = os.environ.get("HF_HOME", DEFAULT_HF_HOME)
os.environ["HF_HOME"] = HF_HOME

# Local model directory names under: <BASE_PATH>/LLMs/<MODEL_CONFIGS[model_key]>
MODEL_CONFIGS: dict[str, str] = {
    "1B": "Llama-3.2-1B-Instruct",
    "3B": "Llama-3.2-3B-Instruct",
    "8B": "Llama-3.1-8B-Instruct",
    "20B": "gpt-oss-20b",
    "27B": "medgemma-27b-text-it",
    "70B": "Llama-3.3-70B-Instruct",
}

# Tokenizer sources (Hugging Face identifiers).
TOKENIZER_HUB: dict[str, str] = {
    "1B": "meta-llama/Llama-3.2-1B-Instruct",
    "3B": "meta-llama/Llama-3.2-3B-Instruct",
    "8B": "meta-llama/Llama-3.1-8B-Instruct",
    "20B": "openai/gpt-oss-20b",
    "27B": "google/medgemma-27b-text-it",
    "70B": "meta-llama/Llama-3.3-70B-Instruct",
}

# Downstream evaluation labels.
OUTPUT_LABELS: list[str] = ["Smoker", "Never Smoker", "Unknown"]

def print_gpu_info():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for gpu_idx in range(num_gpus):
            total_mem = torch.cuda.get_device_properties(gpu_idx).total_memory
            reserved_mem = torch.cuda.memory_reserved(gpu_idx)
            allocated_mem = torch.cuda.memory_allocated(gpu_idx)
            free_mem = reserved_mem - allocated_mem            
            print(f"Device: {torch.cuda.get_device_name(gpu_idx)} (index: {gpu_idx})")
            print(f"Total memory: {total_mem / 1024**3:.2f} GB")
            print(f"Reserved memory: {reserved_mem / 1024**3:.2f} GB")
            print(f"Allocated memory: {allocated_mem / 1024**3:.2f} GB")
            print(f"Free memory within reserved: {free_mem / 1024**3:.2f} GB")
            print("\n" + "-"*40 + "\n")
    else:
        print("Sadly GPU poor :(")
        
# -------------------------
# Stage 1 loader
# -------------------------

def load_stage1_model_and_tokenizer(model_key: str) -> Tuple[Any, Any]:
    """
    Load the Stage 1 model and tokenizer for unstructured classification.

    Parameters
    ----------
    model_key:
        One of the keys in MODEL_CONFIGS and TOKENIZER_HUB.

    Returns
    -------
    (model, tokenizer)
        A Hugging Face model and tokenizer.
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key: {model_key}. Valid keys: {sorted(MODEL_CONFIGS)}")

    model_path = os.path.join(BASE_PATH, "LLMs", MODEL_CONFIGS[model_key])

    # Tokenizer load: keep the current fallback behavior for 20B.
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_HUB[model_key])
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            "openai/gpt-oss-20b",
            use_fast=True,
            cache_dir=os.path.join(BASE_PATH, "LLMs", "gpt-oss-20b"),
        )

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is not None and eos_token_id is not None and pad_token_id == eos_token_id:
        raise ValueError(
            "Pad token and EOS token must be different for clarity in multi-class classification."
        )

    # Quantization and model-family specific branches
    if model_key == "70B":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        common_kwargs = dict(
            device_map="auto",
            quantization_config=bnb_cfg,
        )
        try:
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                cache_dir=os.path.join(BASE_PATH, "LLMs", "models"),
                **common_kwargs,
            )
            print("Loaded 70B model with 4-bit quantization from local path.")
        except Exception:
            model = LlamaForCausalLM.from_pretrained(
                "meta-llama/Llama-3.3-70B-Instruct",
                cache_dir=os.path.join(BASE_PATH, "LLMs", "models"),
                **common_kwargs,
            )
            print("Loaded 70B model with 4-bit quantization from Hugging Face hub.")
    
    elif model_key == "20B":
        # gpt-oss-20b fallback with MXFP4 quantization config
        quant_config = Mxfp4Config(
            torch_dtype="auto",
            device_map="auto",
        )
        model = GptOssForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            dtype="auto",
            device_map="auto",
            quantization_config=quant_config,
            cache_dir=os.path.join(BASE_PATH, "LLMs", "gpt-oss-20b"),
        )
        print("Loaded 20B model with MXFP4 quantization from Hugging Face hub.")

    elif model_key == "27B":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=os.path.join(BASE_PATH, "LLMs", "models"),
        )

    else:
        # Most Llama models (1B, 3B, 8B) load here.
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=os.path.join(BASE_PATH, "LLMs", "models"),
        )
        print(f"Loaded {model_key} model from local path.")
    print_gpu_info()
    return model, tokenizer


# -------------------------
# Stage 2 loader (constrained decoding)
# -------------------------

def load_stage2_structured_generator(
    model_key: str = "8B",
) -> Tuple[Any, Any]:
    """
    Load the Stage 2 constrained decoding generator (Outlines) and tokenizer.

    Stage 2 converts Stage 1 raw outputs into a structured label via constrained
    decoding. The default model_key matches the current project behavior.

    Parameters
    ----------
    model_key:
        Model key to use for constrained decoding. Default is "8B".

    Returns
    -------
    (generator, tokenizer)
        generator is the Outlines wrapper created by from_transformers.
    """
    if from_transformers is None:
        raise ImportError(
            "Outlines is not available. Install outlines to use Stage 2 constrained decoding."
        )

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key: {model_key}. Valid keys: {sorted(MODEL_CONFIGS)}")

    model_path = os.path.join(BASE_PATH, "LLMs", MODEL_CONFIGS[model_key])

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_HUB[model_key])
    llm = LlamaForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=os.path.join(BASE_PATH, "LLMs", "models"),
    )

    generator = from_transformers(llm, tokenizer)

    print_gpu_info()
    return generator, tokenizer
