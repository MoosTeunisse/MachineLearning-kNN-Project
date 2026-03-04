from knnbench.baseline_sklearn import run_sklearn_knn_breast_cancer, run_sklearn_knn_adult  # type: ignore

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    print("\n=== Baseline kNN (sklearn): Adult dataset ===")
    ks = [3, 5, 7, 9, 11]
    scalings = [None, "standard", "minmax"]
    weights_list = ["uniform", "distance"]

    for k in ks:
        for scaling in scalings:
            for weights in weights_list:
                res = run_sklearn_knn_adult(k=k, scaling=scaling, weights=weights)
                print(
                    f"k={k}, scaling={scaling}, weights={weights} -> "
                    f"accuracy={res['accuracy']:.4f}, macro_recall={res['macro_recall']:.4f}, macro_f1={res['macro_f1']:.4f}"
                )


if __name__ == "__main__":
    main()
