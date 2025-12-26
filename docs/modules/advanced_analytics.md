### `advanced_analytics.py` (bootstrap CIs and multiclass evaluation metrics)

Reusable analytics utilities for computing bootstrap confidence intervals and multiclass evaluation metrics. This module is intentionally independent of model inference and file I/O orchestration. It is used downstream of Stage 2 when computing predictive performance metrics, and it can also be used for Stage 1 efficiency summaries (time per note).

This file should remain unchanged. :contentReference[oaicite:0]{index=0}

#### Dependencies

- `numpy`
- `pandas`
- `scikit-learn` (`accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`) :contentReference[oaicite:1]{index=1}

---

#### Functions

- `bootstrap_metric_ci(metric_fn, y_true, y_pred, n_iterations: int = 1000, alpha: float = 0.05) -> tuple[float, tuple[float, float]]`  
  Bootstraps a scalar metric over `(y_true, y_pred)` and returns the mean and percentile confidence interval. :contentReference[oaicite:2]{index=2}

  **Args**
  - `metric_fn`  
    Function accepting `(y_true, y_pred)` and returning a scalar.
  - `y_true`  
    True labels (indexable array-like).
  - `y_pred`  
    Predicted labels (indexable array-like).
  - `n_iterations: int`  
    Number of bootstrap resamples.
  - `alpha: float`  
    Significance level for the CI (0.05 gives a 95% CI).

  **Returns**
  - `(mean_score, (ci_lower, ci_upper))`

---

- `bootstrap_efficiency_metrics(efficiency_values, n_iterations: int = 1000, alpha: float = 0.05) -> tuple[float, tuple[float, float]]`  
  Bootstraps per note efficiency values (for example, `time_seconds`) and returns the mean and percentile confidence interval. Handles empty input and NaNs. :contentReference[oaicite:3]{index=3}

  **Args**
  - `efficiency_values`  
    Array-like of per note values.
  - `n_iterations: int`
  - `alpha: float`

  **Returns**
  - `(mean_value, (ci_lower, ci_upper))`

---

- `compute_spec_npv(yt, yp, labels: list[str]) -> dict`  
  Computes specificity and NPV (negative predictive value) from the confusion matrix, returning per-class values and macro, micro, and weighted aggregates. :contentReference[oaicite:4]{index=4}

  **Args**
  - `yt`  
    True labels.
  - `yp`  
    Predicted labels.
  - `labels: list[str]`  
    Ordered class labels to use for the confusion matrix.

  **Returns**
  - `dict` with keys:
    - `per_class_spec`, `per_class_npv`
    - `macro_spec`, `micro_spec`, `weighted_spec`
    - `macro_npv`, `micro_npv`, `weighted_npv`

---

- `compute_per_class_clinical_metrics(yt, yp, labels: list[str]) -> pd.DataFrame`  
  Computes per-class clinical metrics with bootstrap CIs:
  - PPV (precision)
  - Sensitivity (recall)
  - Specificity
  - NPV  
  Returns a dataframe with one row per class. :contentReference[oaicite:5]{index=5}

  **Args**
  - `yt`
  - `yp`
  - `labels: list[str]`

  **Returns**
  - `pd.DataFrame` with columns:
    - `Class`
    - `PPV`, `PPV_CI_Lower`, `PPV_CI_Upper`
    - `Sensitivity`, `Sensitivity_CI_Lower`, `Sensitivity_CI_Upper`
    - `Specificity`, `Specificity_CI_Lower`, `Specificity_CI_Upper`
    - `NPV`, `NPV_CI_Lower`, `NPV_CI_Upper`

---

- `compute_comprehensive_metrics(y_true, y_pred, output_labels: list[str]) -> dict`  
  Computes a comprehensive suite of multiclass performance metrics with bootstrap CIs, including:
  - Accuracy
  - Macro, micro, weighted precision, recall, F1
  - Clinical aliases: PPV and sensitivity (precision and recall)
  - Specificity and NPV (macro, micro, weighted), each with bootstrap CIs :contentReference[oaicite:6]{index=6}

  **Args**
  - `y_true`
  - `y_pred`
  - `output_labels: list[str]`

  **Returns**
  - `dict` mapping metric names to scalar values and CI bounds (for example `Macro_F1`, `Macro_F1_CI_Lower`, `Macro_F1_CI_Upper`).

---

- `save_confusion_matrix(y_true, y_pred, output_labels: list[str], tag: str, results_dir: str = "results/conf_mats") -> pd.DataFrame`  
  Writes a confusion matrix CSV with a one-line header explanation, using `tag` to name the file. Returns the raw confusion matrix array. :contentReference[oaicite:7]{index=7}

  **Args**
  - `y_true`
  - `y_pred`
  - `output_labels: list[str]`
  - `tag: str`  
    Identifier used in the output filename.
  - `results_dir: str`  
    Directory to write confusion matrices.

  **Returns**
  - Confusion matrix array (as returned by `sklearn.metrics.confusion_matrix`).
