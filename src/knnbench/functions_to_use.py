from itertools import product
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from experiments.manual_knn import ManualKNNClassifier
from knnbench.datasets import load_adult_df, preprocess_adult_df, transform_adult_df
from knnbench.utils import compute_metrics, set_seed

def run_manual_one_adult(
    *,
    k: int,
    scaling: str | None,
    voting: str,
    tie_break: str,
    seed: int = 42,
    batch_size: int = 512,
) -> dict[str, Any]:
    """
    Run one manual kNN configuration on Adult and return:
    - the configuratinon (k, scaling, voting, tie_break, seed, batch_size)
    - metrics
    - confusion matrix + labels
    - ties
    """
    set_seed(seed)

    X, y = load_adult_df()
    if "fnlwgt" in X.columns:
        X = X.drop(columns=["fnlwgt"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    X_train_proc, prep = preprocess_adult_df(X_train, scaling=scaling, return_preprocessor=True)
    X_test_proc = transform_adult_df(X_test, prep)

    knn = ManualKNNClassifier(k=k, voting=voting, tie_break=tie_break).fit(X_train_proc, y_train)
    y_pred = knn.predict(X_test_proc, batch_size=batch_size)

    metrics = compute_metrics(y_test, y_pred)

    labels = ["<=50K", ">50K"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    return {
        "config": dict(
            model="manual",
            k=k,
            scaling=scaling,
            voting=voting,
            tie_break=tie_break,
            seed=seed,
            batch_size=batch_size,
        ),
        "metrics": metrics,
        "confusion_matrix": cm,
        "labels": labels,
        "ties": getattr(knn, "last_num_ties_", None),
    }

def run_manual_grid_adult(
    *,
    ks: list[int],
    scalings: list[str | None] = [None, "standard", "minmax"],
    votings: list[str] = ["uniform", "distance"],
    tie_breaks: list[str] = ["nearest", "min_class"],
    seed: int = 42,
    batch_size: int = 512,
    include_ties: bool = True,
    val_size = 0.25,
) -> list[dict[str, Any]]:
    """
    Run many manual kNN configurations and return a list of rows with only:
    - accuracy, macro_recall, macro_f1
    - ties (if include_ties is True)
    
    Workflow:
    1. split once into train_full / test, we don't use test until the end
    2. split train_full into train / val
    3. Tune configs on validation set only
    4. Return validation restults only
    """
    set_seed(seed)

    X, y = load_adult_df()
    if "fnlwgt" in X.columns:
        X = X.drop(columns=["fnlwgt"])

    # Split ONCE for fairness + speed
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=seed, stratify=y_train_full
    )

    rows: list[dict[str, Any]] = []

    for scaling in scalings:
        X_train_proc, prep = preprocess_adult_df(X_train, scaling=scaling, return_preprocessor=True)
        X_val_proc = transform_adult_df(X_val, prep)

        for k in ks:
            for voting, tie_break in product(votings, tie_breaks):
                knn = ManualKNNClassifier(k=k, voting=voting, tie_break=tie_break).fit(X_train_proc, y_train)
                
                y_pred = knn.predict(X_val_proc, batch_size=batch_size)
                m = compute_metrics(y_val, y_pred)

                row = {
                    "model": "manual",
                    "split": "val",
                    "k": k,
                    "scaling": scaling,
                    "weights": voting,
                    "tie_break": tie_break,
                    "accuracy": float(m["accuracy"]),
                    "macro_precision": float(m["macro_precision"]),
                    "macro_recall": float(m["macro_recall"]),
                    "macro_f1": float(m["macro_f1"]),
                }
                if include_ties:
                    row["ties"] = getattr(knn, "last_num_ties_", None)

                rows.append(row)

    return rows

def run_sklearn_grid_adult(
    *,
    ks: list[int],
    scalings: list[str | None] = [None, "standard", "minmax"],
    weights_list: list[str] = ["uniform", "distance"],
    seed: int = 42,
    val_size = 0.25,
) -> list[dict[str, Any]]:
    """
    Run many sklearn KNN configs and return a list of rows with only:
    - accuracy, macro_recall, macro_f1
    
    Workflow:
    1. split once into train_full / test, we don't use test until the end
    2. split train_full into train / val
    3. Tune configs on validation set only
    4. Return validation restults only
    """
    set_seed(seed)

    X, y = load_adult_df()
    if "fnlwgt" in X.columns:
        X = X.drop(columns=["fnlwgt"])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=seed, stratify=y_train_full
    )

    rows: list[dict[str, Any]] = []

    for scaling in scalings:
        X_train_proc, prep = preprocess_adult_df(X_train, scaling=scaling, return_preprocessor=True)
        X_val_proc = transform_adult_df(X_val, prep)

        for k in ks:
            for weights in weights_list:
                clf = KNeighborsClassifier(n_neighbors=k, weights=weights)
                clf.fit(X_train_proc, y_train)
                y_pred = clf.predict(X_val_proc)
                m = compute_metrics(y_val, y_pred)

                rows.append(
                    {
                        "model": "sklearn",
                        "split": "val",
                        "k": k,
                        "scaling": scaling,
                        "weights": weights,
                        "tie_break": None,
                        "accuracy": float(m["accuracy"]),
                        "macro_precision": float(m["macro_precision"]),
                        "macro_recall": float(m["macro_recall"]),
                        "macro_f1": float(m["macro_f1"]),
                    }
                )

    return rows

def compare_manual_vs_sklearn_adult(
    *,
    ks: list[int],
    scalings: list[str | None] = [None, "standard", "minmax"],
    weights_list: list[str] = ["uniform", "distance"],
    seed: int = 42,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    """
    Compare manual vs sklearn under the same settings.
    Use manual tie_break='min_class' to mirror argmax default behavior.
    Returns rows with both metrics and deltas.
    """
    manual_rows = run_manual_grid_adult(
        ks=ks,
        scalings=scalings,
        votings=weights_list,
        tie_breaks=["min_class"],
        seed=seed,
        batch_size=batch_size,
        include_ties=False,
    )
    sk_rows = run_sklearn_grid_adult(
        ks=ks,
        scalings=scalings,
        weights_list=weights_list,
        seed=seed,
    )

    sk_map = {(r["k"], r["scaling"], r["weights"]): r for r in sk_rows}

    out: list[dict[str, Any]] = []
    for mr in manual_rows:
        key = (mr["k"], mr["scaling"], mr["weights"])
        sr = sk_map.get(key)
        if sr is None:
            continue

        out.append(
            {
                "k": mr["k"],
                "scaling": mr["scaling"],
                "weights": mr["weights"],
                "accuracy_manual": mr["accuracy"],
                "accuracy_sklearn": sr["accuracy"],
                "d_accuracy": mr["accuracy"] - sr["accuracy"],
                "macro_recall_manual": mr["macro_recall"],
                "macro_recall_sklearn": sr["macro_recall"],
                "d_macro_recall": mr["macro_recall"] - sr["macro_recall"],
                "macro_f1_manual": mr["macro_f1"],
                "macro_f1_sklearn": sr["macro_f1"],
                "d_macro_f1": mr["macro_f1"] - sr["macro_f1"],
            }
        )

    return out

def select_best_config(
    rows: list[dict[str, Any]],
    metric: str = "macro_f1",
) -> dict[str, Any]:
    """
    Select the best config based on the given metric (default "macro_f1").
    """
    best_row = max(rows, key=lambda r: r[metric])
    return best_row

def run_manual_best_on_test_adult(
    *,
    k: int,
    scaling: str | None,
    voting: str,
    tie_break: str,
    seed: int = 42,
    batch_size: int = 512,
) -> dict[str, Any]:
    """
    Run the best manual kNN configuration on the test set and return metrics + confusion matrix.
    """
    set_seed(seed)

    X, y = load_adult_df()
    if "fnlwgt" in X.columns:
        X = X.drop(columns=["fnlwgt"])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    X_train_full_proc, prep = preprocess_adult_df(X_train_full, scaling=scaling, return_preprocessor=True)
    X_test_proc = transform_adult_df(X_test, prep)
    
    knn = ManualKNNClassifier(k=k, voting=voting, tie_break=tie_break).fit(X_train_full_proc, y_train_full)
    y_pred = knn.predict(X_test_proc, batch_size=batch_size)
    metrics = compute_metrics(y_test, y_pred)
    
    return {
        "model": "manual",
        "split": "test",
        "k": k,
        "scaling": scaling,
        "weights": voting,
        "tie_break": tie_break,
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "ties": getattr(knn, "last_num_ties_", None),
    }

def run_sklearn_best_on_test_adult(
    *,
    k: int,
    scaling: str | None,
    weights: str,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run the best sklearn KNN configuration on the test set and return metrics + confusion matrix.
    """
    set_seed(seed)

    X, y = load_adult_df()
    if "fnlwgt" in X.columns:
        X = X.drop(columns=["fnlwgt"])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    X_train_full_proc, prep = preprocess_adult_df(X_train_full, scaling=scaling, return_preprocessor=True)
    X_test_proc = transform_adult_df(X_test, prep)
    
    clf = KNeighborsClassifier(n_neighbors=k, weights=weights)
    clf.fit(X_train_full_proc, y_train_full)
    
    y_pred = clf.predict(X_test_proc)
    metrics = compute_metrics(y_test, y_pred)
    
    return {
        "model": "sklearn",
        "split": "test",
        "k": k,
        "scaling": scaling,
        "weights": weights,
        "tie_break": None,
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
    }