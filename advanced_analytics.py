"""
Advanced analytics module for bootstrap confidence intervals and multiclass metrics.

This module extracts the complex bootstrap and multiclass analysis functionality
from the main evaluation pipeline to improve code organization and maintainability.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def bootstrap_metric_ci(metric_fn, y_true, y_pred, n_iterations=1000, alpha=0.05):
    """
    Calculate bootstrap confidence intervals for a given metric function.
    
    Args:
        metric_fn: Function that takes (y_true, y_pred) and returns a scalar metric
        y_true: True labels
        y_pred: Predicted labels  
        n_iterations: Number of bootstrap iterations (default: 1000)
        alpha: Significance level for CI (default: 0.05 for 95% CI)
        
    Returns:
        tuple: (mean_score, (lower_bound, upper_bound))
    """
    scores = []
    for _ in range(n_iterations):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        score = metric_fn(sample_true, sample_pred)
        scores.append(score)
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return np.mean(scores), (lower, upper)


def bootstrap_efficiency_metrics(efficiency_values, n_iterations=1000, alpha=0.05):
    """
    Calculate bootstrap confidence intervals for efficiency metrics (time/VRAM per note).
    
    Args:
        efficiency_values: Array of per-note efficiency values
        n_iterations: Number of bootstrap iterations (default: 1000)
        alpha: Significance level for CI (default: 0.05 for 95% CI)
        
    Returns:
        tuple: (mean_value, (lower_bound, upper_bound))
    """
    if len(efficiency_values) == 0:
        return float("nan"), (float("nan"), float("nan"))
    
    # Remove NaN values for bootstrap sampling
    clean_values = np.array([v for v in efficiency_values if not np.isnan(v)])
    if len(clean_values) == 0:
        return float("nan"), (float("nan"), float("nan"))
    
    bootstrap_means = []
    for _ in range(n_iterations):
        sample = np.random.choice(clean_values, size=len(clean_values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return np.mean(clean_values), (lower, upper)


def compute_spec_npv(yt, yp, labels):
    """
    Compute specificity and negative predictive value (NPV) metrics.
    
    Args:
        yt: True labels
        yp: Predicted labels
        labels: List of class labels
        
    Returns:
        dict: Dictionary containing per-class, macro, micro, and weighted metrics
    """
    cm = confusion_matrix(yt, yp, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    # Per-class
    with np.errstate(divide="ignore", invalid="ignore"):
        spec_pc = np.where((tn + fp) > 0, tn / (tn + fp), np.nan)
        npv_pc  = np.where((tn + fn) > 0, tn / (tn + fn), np.nan)

    support = cm.sum(axis=1).astype(float)

    # Macro
    macro_spec = np.nanmean(spec_pc)
    macro_npv  = np.nanmean(npv_pc)

    # Weighted by true-class support
    sw = np.nansum(support)
    weighted_spec = np.nansum(spec_pc * support) / sw if sw > 0 else np.nan
    weighted_npv  = np.nansum(npv_pc  * support) / sw if sw > 0 else np.nan

    # Micro from pooled counts
    micro_tn = tn.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_spec = micro_tn / (micro_tn + micro_fp) if (micro_tn + micro_fp) > 0 else np.nan
    micro_npv  = micro_tn / (micro_tn + micro_fn) if (micro_tn + micro_fn) > 0 else np.nan

    return {
        "per_class_spec": spec_pc,
        "per_class_npv":  npv_pc,
        "macro_spec": macro_spec,
        "weighted_spec": weighted_spec,
        "micro_spec": micro_spec,
        "macro_npv": macro_npv,
        "weighted_npv": weighted_npv,
        "micro_npv": micro_npv,
    }


def compute_per_class_clinical_metrics(yt, yp, labels):
    """
    Compute per-class clinical metrics (PPV, Sensitivity, Specificity, NPV) with bootstrap CIs.
    
    Args:
        yt: True labels
        yp: Predicted labels
        labels: List of class labels
        
    Returns:
        pd.DataFrame: Per-class metrics with confidence intervals
    """
    # PPV and Sensitivity per class come from precision_recall_fscore_support
    prec_pc, rec_pc, _, _ = precision_recall_fscore_support(
        yt, yp, labels=labels, average=None, zero_division=0
    )
    extras = compute_spec_npv(yt, yp, labels)
    rows = []
    
    for i, lab in enumerate(labels):
        # Bootstrap CIs for each class
        ppv_ci = bootstrap_metric_ci(
            lambda a, b, i=i: precision_recall_fscore_support(
                a, b, labels=labels, average=None, zero_division=0
            )[0][i],
            yt, yp
        )[1]
        sens_ci = bootstrap_metric_ci(
            lambda a, b, i=i: precision_recall_fscore_support(
                a, b, labels=labels, average=None, zero_division=0
            )[1][i],
            yt, yp
        )[1]
        spec_ci = bootstrap_metric_ci(
            lambda a, b, i=i: compute_spec_npv(a, b, labels)["per_class_spec"][i],
            yt, yp
        )[1]
        npv_ci = bootstrap_metric_ci(
            lambda a, b, i=i: compute_spec_npv(a, b, labels)["per_class_npv"][i],
            yt, yp
        )[1]

        rows.append({
            "Class": lab,
            "PPV": float(prec_pc[i]), "PPV_CI_Lower": ppv_ci[0], "PPV_CI_Upper": ppv_ci[1],
            "Sensitivity": float(rec_pc[i]), "Sensitivity_CI_Lower": sens_ci[0], "Sensitivity_CI_Upper": sens_ci[1],
            "Specificity": float(extras["per_class_spec"][i]), "Specificity_CI_Lower": spec_ci[0], "Specificity_CI_Upper": spec_ci[1],
            "NPV": float(extras["per_class_npv"][i]), "NPV_CI_Lower": npv_ci[0], "NPV_CI_Upper": npv_ci[1],
        })
    return pd.DataFrame(rows)


def compute_comprehensive_metrics(y_true, y_pred, output_labels):
    """
    Compute comprehensive evaluation metrics with bootstrap confidence intervals.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_labels: List of class labels
        
    Returns:
        dict: Dictionary containing all metrics with confidence intervals
    """
    # Accuracy
    acc, acc_ci = bootstrap_metric_ci(accuracy_score, y_true, y_pred)

    # Macro metrics
    macro_prec, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_prec_ci = bootstrap_metric_ci(
        lambda yt, yp: precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)[0],
        y_true, y_pred
    )[1]
    macro_recall_ci = bootstrap_metric_ci(
        lambda yt, yp: precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)[1],
        y_true, y_pred
    )[1]
    macro_f1_ci = bootstrap_metric_ci(
        lambda yt, yp: precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)[2],
        y_true, y_pred
    )[1]

    # Micro metrics
    micro_prec, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    micro_prec_ci = bootstrap_metric_ci(
        lambda yt, yp: precision_recall_fscore_support(yt, yp, average="micro", zero_division=0)[0],
        y_true, y_pred
    )[1]
    micro_recall_ci = bootstrap_metric_ci(
        lambda yt, yp: precision_recall_fscore_support(yt, yp, average="micro", zero_division=0)[1],
        y_true, y_pred
    )[1]
    micro_f1_ci = bootstrap_metric_ci(
        lambda yt, yp: precision_recall_fscore_support(yt, yp, average="micro", zero_division=0)[2],
        y_true, y_pred
    )[1]

    # Weighted metrics
    weighted_prec, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    weighted_prec_ci = bootstrap_metric_ci(
        lambda yt, yp: precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)[0],
        y_true, y_pred
    )[1]
    weighted_recall_ci = bootstrap_metric_ci(
        lambda yt, yp: precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)[1],
        y_true, y_pred
    )[1]
    weighted_f1_ci = bootstrap_metric_ci(
        lambda yt, yp: precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)[2],
        y_true, y_pred
    )[1]

    # Specificity and NPV with CIs
    spec_npv = compute_spec_npv(y_true, y_pred, output_labels)

    macro_spec_ci = bootstrap_metric_ci(
        lambda yt, yp: compute_spec_npv(yt, yp, output_labels)["macro_spec"],
        y_true, y_pred
    )[1]
    micro_spec_ci = bootstrap_metric_ci(
        lambda yt, yp: compute_spec_npv(yt, yp, output_labels)["micro_spec"],
        y_true, y_pred
    )[1]
    weighted_spec_ci = bootstrap_metric_ci(
        lambda yt, yp: compute_spec_npv(yt, yp, output_labels)["weighted_spec"],
        y_true, y_pred
    )[1]

    macro_npv_ci = bootstrap_metric_ci(
        lambda yt, yp: compute_spec_npv(yt, yp, output_labels)["macro_npv"],
        y_true, y_pred
    )[1]
    micro_npv_ci = bootstrap_metric_ci(
        lambda yt, yp: compute_spec_npv(yt, yp, output_labels)["micro_npv"],
        y_true, y_pred
    )[1]
    weighted_npv_ci = bootstrap_metric_ci(
        lambda yt, yp: compute_spec_npv(yt, yp, output_labels)["weighted_npv"],
        y_true, y_pred
    )[1]

    return {
        "Accuracy": acc, "Accuracy_CI_Lower": acc_ci[0], "Accuracy_CI_Upper": acc_ci[1],

        # Core precision/recall/F1
        "Macro_Precision": macro_prec, "Macro_Precision_CI_Lower": macro_prec_ci[0], "Macro_Precision_CI_Upper": macro_prec_ci[1],
        "Macro_Recall": macro_recall, "Macro_Recall_CI_Lower": macro_recall_ci[0], "Macro_Recall_CI_Upper": macro_recall_ci[1],
        "Macro_F1": macro_f1, "Macro_F1_CI_Lower": macro_f1_ci[0], "Macro_F1_CI_Upper": macro_f1_ci[1],

        "Micro_Precision": micro_prec, "Micro_Precision_CI_Lower": micro_prec_ci[0], "Micro_Precision_CI_Upper": micro_prec_ci[1],
        "Micro_Recall": micro_recall, "Micro_Recall_CI_Lower": micro_recall_ci[0], "Micro_Recall_CI_Upper": micro_recall_ci[1],
        "Micro_F1": micro_f1, "Micro_F1_CI_Lower": micro_f1_ci[0], "Micro_F1_CI_Upper": micro_f1_ci[1],

        "Weighted_Precision": weighted_prec, "Weighted_Precision_CI_Lower": weighted_prec_ci[0], "Weighted_Precision_CI_Upper": weighted_prec_ci[1],
        "Weighted_Recall": weighted_recall, "Weighted_Recall_CI_Lower": weighted_recall_ci[0], "Weighted_Recall_CI_Upper": weighted_recall_ci[1],
        "Weighted_F1": weighted_f1, "Weighted_F1_CI_Lower": weighted_f1_ci[0], "Weighted_F1_CI_Upper": weighted_f1_ci[1],

        # Clinical aliases (PPV == Precision, Sensitivity == Recall)
        "Macro_PPV": macro_prec, "Macro_PPV_CI_Lower": macro_prec_ci[0], "Macro_PPV_CI_Upper": macro_prec_ci[1],
        "Micro_PPV": micro_prec, "Micro_PPV_CI_Lower": micro_prec_ci[0], "Micro_PPV_CI_Upper": micro_prec_ci[1],
        "Weighted_PPV": weighted_prec, "Weighted_PPV_CI_Lower": weighted_prec_ci[0], "Weighted_PPV_CI_Upper": weighted_prec_ci[1],

        "Macro_Sensitivity": macro_recall, "Macro_Sensitivity_CI_Lower": macro_recall_ci[0], "Macro_Sensitivity_CI_Upper": macro_recall_ci[1],
        "Micro_Sensitivity": micro_recall, "Micro_Sensitivity_CI_Lower": micro_recall_ci[0], "Micro_Sensitivity_CI_Upper": micro_recall_ci[1],
        "Weighted_Sensitivity": weighted_recall, "Weighted_Sensitivity_CI_Lower": weighted_recall_ci[0], "Weighted_Sensitivity_CI_Upper": weighted_recall_ci[1],

        # Specificity and NPV
        "Macro_Specificity": spec_npv["macro_spec"], "Macro_Specificity_CI_Lower": macro_spec_ci[0], "Macro_Specificity_CI_Upper": macro_spec_ci[1],
        "Micro_Specificity": spec_npv["micro_spec"], "Micro_Specificity_CI_Lower": micro_spec_ci[0], "Micro_Specificity_CI_Upper": micro_spec_ci[1],
        "Weighted_Specificity": spec_npv["weighted_spec"], "Weighted_Specificity_CI_Lower": weighted_spec_ci[0], "Weighted_Specificity_CI_Upper": weighted_spec_ci[1],

        "Macro_NPV": spec_npv["macro_npv"], "Macro_NPV_CI_Lower": macro_npv_ci[0], "Macro_NPV_CI_Upper": macro_npv_ci[1],
        "Micro_NPV": spec_npv["micro_npv"], "Micro_NPV_CI_Lower": micro_npv_ci[0], "Micro_NPV_CI_Upper": micro_npv_ci[1],
        "Weighted_NPV": spec_npv["weighted_npv"], "Weighted_NPV_CI_Lower": weighted_npv_ci[0], "Weighted_NPV_CI_Upper": weighted_npv_ci[1],
    }


def save_confusion_matrix(y_true, y_pred, output_labels, tag, results_dir="results/conf_mats"):
    """
    Save confusion matrix with explanation header.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_labels: List of class labels
        tag: Identifier for the model/temperature combination
        results_dir: Directory to save confusion matrices
    """
    import os
    
    os.makedirs(results_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=output_labels)
    cm_df = pd.DataFrame(cm, index=output_labels, columns=output_labels)
    explanation = pd.DataFrame({output_labels[0]: ["Rows = True labels, Columns = Predicted labels"]})
    with open(f"{results_dir}/confusion_matrix_{tag}.csv", "w", newline="") as f:
        explanation.to_csv(f, header=False, index=False)
        cm_df.to_csv(f)
    
    return cm