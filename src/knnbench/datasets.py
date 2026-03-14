import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.datasets import fetch_openml
from sklearn.datasets import load_breast_cancer

def load_breast_cancer_df():
    """Load data frame with X is features and y is target."""
    
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y

def load_adult_df(raw: bool = True):
    """Load Adult dataset (census income) from OpenML.
    If raw is True, return the raw data frame.
    If raw is False, return the preprocessed data frame with one-hot encoding.
    """
    #load the data
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    df = adult.frame.copy()
    
    #find the target column (in adult it is just called "class", but in other datasets it may be different)
    if "class" in df.columns:
        target_col = "class"
    else:
        target_col = df.columns[-1]
    
    #designate target and features
    y = df[target_col].astype(str).str.strip()
    X = df.drop(columns=[target_col])
    X = X.drop(columns=["fnlwgt", "education"], errors='ignore') #since we already have education-num, and fnlwgt is not a meaningful feature for prediction
    
    X = X.replace('?', np.nan)
    
    return X, y

def preprocess_adult_df(X, scaling=None, return_preprocessor=False):
    """
    X: from the load_adult_df function, the features of the adult dataset.
    scaling: None, "standard", or "minmax".
    return_preprocessor: if true, also return the fitted transformer.
    """
    #split numeric and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in X.columns if col not in num_cols]
    
    #choose scaling
    if scaling is None:
        scaler = None
    elif scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Invalid scaling option: {scaling}, must be None, 'standard', or 'minmax'.")
    
    if scaler == "passthrough":
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy='median'))
        ])
    else:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", scaler)
        ])
    
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])
    
    #fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    if return_preprocessor:
        return X_processed, preprocessor
    
    return X_processed

def transform_adult_df(X, preprocessor):
    """Transform the adult dataset with a fitted preprocessor."""
    return preprocessor.transform(X)