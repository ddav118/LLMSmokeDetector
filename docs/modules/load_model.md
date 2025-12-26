### `load_model.py` (model and tokenizer loading)

Single source of truth for loading all models and tokenizers used in this project, including Stage 1 inference and Stage 2 constrained decoding. This module also prints GPU device memory info after each load via `print_gpu_info()`.

#### Constants

- `BASE_PATH: str`  
  Base directory containing local model weights.

- `MODEL_CONFIGS: dict[str, str]`  
  Maps model keys to local model directory names (Stage 1).

- `TOKENIZER_HUB: dict[str, str]`  
  Maps model keys to tokenizer identifiers (Stage 1).

- `OUTPUT_LABELS: list[str]`  
  Allowed structured smoking status labels used for evaluation.

#### Functions

- `load_stage1_model_and_tokenizer(model_key: str)`  
  Loads the Stage 1 LLM and tokenizer used for initial unstructured classification.

  **Args**
  - `model_key: str`  
    Model key, one of `1B | 3B | 8B | 20B | 27B | 70B`.

  **Returns**
  - `(model, tokenizer)`  
    Loaded model and tokenizer objects.

  **Notes**
  - Applies model specific loading and quantization branches as needed.
  - Calls `print_gpu_info()` after loading.

- `load_stage2_structured_generator()`  
  Loads the Stage 2 constrained decoding model and returns a structured generation wrapper.

  **Args**
  - None

  **Returns**
  - `(generator, tokenizer)`  
    `generator` is the constrained decoding callable (for example an Outlines wrapper).

  **Notes**
  - Calls `print_gpu_info()` after loading.### `load_model.py` (model and tokenizer loading)

Single source of truth for loading all models and tokenizers used in this project, including Stage 1 inference and Stage 2 constrained decoding. This module also prints GPU device memory info after each load via `print_gpu_info()`.

#### Constants

- `BASE_PATH: str`  
  Base directory containing local model weights.

- `MODEL_CONFIGS: dict[str, str]`  
  Maps model keys to local model directory names (Stage 1).

- `TOKENIZER_HUB: dict[str, str]`  
  Maps model keys to tokenizer identifiers (Stage 1).

- `OUTPUT_LABELS: list[str]`  
  Allowed structured smoking status labels used for evaluation.

#### Functions

- `load_stage1_model_and_tokenizer(model_key: str)`  
  Loads the Stage 1 LLM and tokenizer used for initial unstructured classification.

  **Args**
  - `model_key: str`  
    Model key, one of `1B | 3B | 8B | 20B | 27B | 70B`.

  **Returns**
  - `(model, tokenizer)`  
    Loaded model and tokenizer objects.

  **Notes**
  - Applies model specific loading and quantization branches as needed.
  - Calls `print_gpu_info()` after loading.

- `load_stage2_structured_generator()`  
  Loads the Stage 2 constrained decoding model and returns a structured generation wrapper.

  **Args**
  - None

  **Returns**
  - `(generator, tokenizer)`  
    `generator` is the constrained decoding callable (for example an Outlines wrapper).

  **Notes**
  - Calls `print_gpu_info()` after loading.