import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def set_seed(seed: int = 42):
    """set the same seed everytime for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def compute_metrics(y_true, y_pred):
    """
    Returns a dict of metrics.
    """
    labels = np.unique(y_true)
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

    precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None)
    recall_per_class = recall_score(y_true, y_pred, labels=labels, average=None)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None)

    for label, p, r, f in zip(labels, precision_per_class, recall_per_class, f1_per_class):
        metrics[f"precision_{label}"] = float(p)
        metrics[f"recall_{label}"] = float(r)
        metrics[f"f1_{label}"] = float(f)
        
    return metrics

def tie_frequency(tie_mask):
    """
    tie_mask is a boolean (it's true if prediction required tie breaking)
    """
    tie_mask = np.asarray(tie_mask, dtype=bool)
    return float(np.mean(tie_mask)) if tie_mask.size > 0 else 0.0

def prediction_disagreement_rate(pred_a, pred_b):
    """
    Fraction of samples where predictions differ.
    """
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    return float(np.mean(pred_a != pred_b))

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))