from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from .datasets import load_breast_cancer_df
from .utils import set_seed, benchmark, peak_memory_bytes

def run_sklearn_knn_benchmark(k: int = 5, test_size: float = 0.2, seed: int = 42):
    """Run kNN from sklearn as a benchmark."""
    set_seed(seed)
    
    #load and split the dataset
    X, y = load_breast_cancer_df()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    
    #standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # train kNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    timing = benchmark(lambda: knn.predict(X_test_scaled))
    peak_bytes = peak_memory_bytes(lambda: knn.predict(X_test_scaled))
    
    return {
        "k": k,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X_train.shape[1],
        "accuracy": float(accuracy),
        "peak_memory_bytes": peak_bytes,
        "peak_memory_mb": peak_bytes / (1024 * 1024),
        **timing,
    }