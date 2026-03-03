import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, recall_score

def set_seed(seed: int = 42):
    """set the same seed everytime for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def compute_metrics(y_true, y_pred):
    """
    Returns a dict of metrics.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, average="macro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

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