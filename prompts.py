"""
Prompt templates for LLMSmokeDetector.

This module centralizes all prompt construction for:
- Stage 1: unstructured smoking status classification from a clinical note
- Stage 2: sanitization / normalization of Stage 1 raw outputs into a structured label

This module must not load models or run inference.
"""

from __future__ import annotations

from typing import Dict, List

try:
    # Preferred: single source of truth
    from load_model import OUTPUT_LABELS  # type: ignore
except Exception:
    # Fallback to keep prompts usable in isolation
    OUTPUT_LABELS = ["Smoker", "Never Smoker", "Unknown"]


Message = Dict[str, str]


def prompt_setup(note_text: str, reasoning_effort=None) -> List[Message]:
    """
    Stage 1 prompt for unstructured smoking status classification.

    Args
    ----
    note_text:
        Full clinical note text.
    reasoning_effort:
        Optional parameter to indicate if using gpt-oss model

    Returns
    -------
    List[Message]:
        Chat messages formatted for the Stage 1 model.
    """
    system = (
        "You are an expert medical classifier. "
        "Your task is to determine a patient’s tobacco smoking status based ONLY on the clinical note. "
    )
    user = f"""
    Task: Classify the patient's tobacco smoking status from the clinical note as one of the labels defined below.

    **Smoker**: Choose this label if the note contains any explicit mention that the patient has smoked tobacco at any time, whether currently or in the past. This includes use of nicotine products but does not include marijuana use.
        Look for phrases such as: "smoker," "tobacco use," "uses nicotine," "former smoker," "quit smoking."

    **Never Smoker**: Choose this label if the note contains an explicit statement that the patient has never smoked tobacco.
        Look for phrases such as: "never smoked," "denies tobacco use".

    **Unknown**: Choose this label if the note contains no information about smoking status. This is the default choice.

    Clinical Note:

    \"\"\"{note_text}\"\"\"
    Question: What is the best classification choice?
    """
    if reasoning_effort:
        return [
            {"role": "developer", "content": system.strip()},
            {"role": "user",   "content": user.strip()}
        ]
    else:
        return [
            {"role": "system", "content": system.strip()},
            {"role": "user",   "content": user.strip()}
        ]


def prompt_sanitize(raw_output: str) -> List[Message]:
    """
    Stage 2 prompt for sanitization / normalization into a structured label.

    Args
    ----
    raw_output:
        Raw unstructured output produced by Stage 1.

    Returns
    -------
    List[Message]:
        Chat messages used for constrained decoding to a single label.
    """
    # Keep the original string content aligned with the existing project behavior.
    system = (
        "You are an expert medical classifier. "
        "Classify tobacco smoking status based ONLY on the provided text. "
    )
    user = f"""
    Task: Classify tobacco smoking status from the text as one of the labels defined below:

    **Smoker**: Choose this label if the text contains any explicit mention that the patient has smoked tobacco at any time, whether currently or in the past. This includes use of nicotine products but does not include marijuana use.
        Look for phrases such as: "smoker," "tobacco use," "uses nicotine," "former smoker," "quit smoking."

    **Never Smoker**: Choose this label if the text contains an explicit statement that the patient has never smoked tobacco.
        Look for phrases such as: "never smoked," "denies tobacco use".

    **Unknown**: Choose this label if the text contains no information about smoking status. This is the default choice.

    \"\"\"{raw_output}\"\"\"
    RESULT:
    """
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user",   "content": user.strip()}
    ]



__all__ = [
    "OUTPUT_LABELS",
    "prompt_setup",
    "prompt_sanitize",
]
