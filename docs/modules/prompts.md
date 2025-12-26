### `prompts.py` (centralized model prompts)

Single source of truth for prompt templates used in:
- Stage 1: unstructured smoking status classification from a clinical note
- Stage 2: sanitization of Stage 1 raw output into a single structured label (for constrained decoding)

This module performs prompt construction only. It does not load models or run inference.

#### Constants

- `OUTPUT_LABELS: list[str]`  
  Allowed structured labels for smoking status. Imported from `load_model.OUTPUT_LABELS` when available, otherwise defaults to `["Smoker", "Never Smoker", "Unknown"]`.

#### Functions

- `prompt_setup(note_text: str, reasoning_effort: str | None = None) -> list[dict]`  
  Stage 1 prompt. Takes the full clinical note text and asks the model to output the best label choice.

  **Args**
  - `note_text: str`  
    Full clinical note text.
  - `reasoning_effort: str | None`  
    If provided, the first instruction message uses the `developer` role instead of `system`. This flag does not change prompt text content.

  **Returns**
  - `messages: list[dict]`  
    Chat messages for Stage 1 inference.

- `prompt_sanitize(raw_output: str) -> list[dict]`  
  Stage 2 prompt. Takes raw unstructured output (from Stage 1) and anchors the completion with `RESULT:` to support constrained decoding to a single label in `OUTPUT_LABELS`.

  **Args**
  - `raw_output: str`  
    Raw unstructured output produced by Stage 1.

  **Returns**
  - `messages: list[dict]`  
    Chat messages for Stage 2 constrained decoding.