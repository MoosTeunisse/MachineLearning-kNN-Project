from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .datasets import load_breast_cancer_df, load_adult_df, preprocess_adult_df, transform_adult_df
from .utils import set_seed, compute_metrics


def run_sklearn_knn_breast_cancer(k=5, test_size=0.2, seed=42):
    """
    Sanity-check baseline on breast cancer dataset.
    Uses standard scaling + uniform voting (common default).
    """
    set_seed(seed)

    X, y = load_breast_cancer_df()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="uniform")),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    out = {"dataset": "breast_cancer", "k": k, "test_size": test_size, "seed": seed}
    out.update(compute_metrics(y_test, y_pred))
    return out


def run_sklearn_knn_adult(k=5, test_size=0.2, seed=42, scaling="standard", weights="uniform"):
    """
    Baseline on Adult dataset with:
      - one-hot encoding for categoricals
      - optional scaling for numeric features
      - sklearn voting choice via weights='uniform' or 'distance'
    """
    set_seed(seed)

    X, y = load_adult_df()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Fit preprocessor on train only, then transform test with same fitted mapping
    X_train_proc, prep = preprocess_adult_df(X_train, scaling=scaling, return_preprocessor=True)
    X_test_proc = transform_adult_df(X_test, prep)

    model = KNeighborsClassifier(n_neighbors=k, weights=weights)
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)

    out = {
        "dataset": "adult",
        "k": k,
        "test_size": test_size,
        "seed": seed,
        "scaling": scaling,
        "weights": weights,
    }
    out.update(compute_metrics(y_test, y_pred))
    return out
