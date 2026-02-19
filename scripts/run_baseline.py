from knnbench.baseline_sklearn import run_sklearn_knn_benchmark # type: ignore

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    results = run_sklearn_knn_benchmark(k=5)
    
    print("Baseline kNN (sklearn) results:")
    for key, value in results.items():
        print(f"{key}: {value}")
        
if __name__ == "__main__":
    main()