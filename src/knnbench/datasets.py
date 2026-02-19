import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_breast_cancer_df():
    """Load data fram with X is features and y is target."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y
