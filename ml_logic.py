from functools import partial

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Optional gradient-boosting backends. They are heavy to install and may be
# absent locally; register their algorithms only when the import succeeds so
# the app still runs. requirements.txt lists all three for deployment.
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Each registry maps a display name -> (estimator_factory, param_spec_dict).
# The factory is called as factory(**tuned_params); use functools.partial to
# bake in fixed kwargs (e.g. silencing a library's logging).

CLASSIFICATION_MODELS = {
    "Logistic Regression": (LogisticRegression, {
        "C": {"type": "slider", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
        "max_iter": {"type": "slider", "min": 100, "max": 2000, "default": 1000, "step": 100},
    }),
    "Decision Tree": (DecisionTreeClassifier, {
        "max_depth": {"type": "slider", "min": 1, "max": 30, "default": 5, "step": 1},
        "min_samples_split": {"type": "slider", "min": 2, "max": 20, "default": 2, "step": 1},
    }),
    "Random Forest": (RandomForestClassifier, {
        "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100, "step": 10},
        "max_depth": {"type": "slider", "min": 1, "max": 30, "default": 5, "step": 1},
    }),
    "SVM": (SVC, {
        "C": {"type": "slider", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
        "kernel": {"type": "select", "options": ["linear", "rbf", "poly"], "default": "rbf"},
    }),
    "KNN": (KNeighborsClassifier, {
        "n_neighbors": {"type": "slider", "min": 1, "max": 30, "default": 5, "step": 1},
    }),
    "Naive Bayes": (GaussianNB, {}),
    "Gradient Boosting": (GradientBoostingClassifier, {
        "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100, "step": 10},
        "learning_rate": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01},
        "max_depth": {"type": "slider", "min": 1, "max": 15, "default": 3, "step": 1},
    }),
}

_BOOST_SLIDERS_TREE = {
    "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100, "step": 10},
    "learning_rate": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01},
    "max_depth": {"type": "slider", "min": 1, "max": 15, "default": 6, "step": 1},
}
_CATBOOST_SLIDERS = {
    "iterations": {"type": "slider", "min": 10, "max": 500, "default": 100, "step": 10},
    "learning_rate": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01},
    "depth": {"type": "slider", "min": 1, "max": 10, "default": 6, "step": 1},
}

if HAS_XGBOOST:
    CLASSIFICATION_MODELS["XGBoost"] = (
        partial(XGBClassifier, verbosity=0, eval_metric="logloss"), dict(_BOOST_SLIDERS_TREE)
    )
if HAS_LIGHTGBM:
    CLASSIFICATION_MODELS["LightGBM"] = (
        partial(LGBMClassifier, verbose=-1),
        {**_BOOST_SLIDERS_TREE, "max_depth": {"type": "slider", "min": -1, "max": 15, "default": -1, "step": 1}},
    )
if HAS_CATBOOST:
    CLASSIFICATION_MODELS["CatBoost"] = (
        partial(CatBoostClassifier, verbose=0), dict(_CATBOOST_SLIDERS)
    )

REGRESSION_MODELS = {
    "Linear Regression": (LinearRegression, {}),
    "Decision Tree": (DecisionTreeRegressor, {
        "max_depth": {"type": "slider", "min": 1, "max": 30, "default": 5, "step": 1},
        "min_samples_split": {"type": "slider", "min": 2, "max": 20, "default": 2, "step": 1},
    }),
    "Random Forest": (RandomForestRegressor, {
        "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100, "step": 10},
        "max_depth": {"type": "slider", "min": 1, "max": 30, "default": 5, "step": 1},
    }),
    "SVM": (SVR, {
        "C": {"type": "slider", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
        "kernel": {"type": "select", "options": ["linear", "rbf", "poly"], "default": "rbf"},
    }),
    "KNN": (KNeighborsRegressor, {
        "n_neighbors": {"type": "slider", "min": 1, "max": 30, "default": 5, "step": 1},
    }),
    "Gradient Boosting": (GradientBoostingRegressor, {
        "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100, "step": 10},
        "learning_rate": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01},
        "max_depth": {"type": "slider", "min": 1, "max": 15, "default": 3, "step": 1},
    }),
}

if HAS_XGBOOST:
    REGRESSION_MODELS["XGBoost"] = (partial(XGBRegressor, verbosity=0), dict(_BOOST_SLIDERS_TREE))
if HAS_LIGHTGBM:
    REGRESSION_MODELS["LightGBM"] = (
        partial(LGBMRegressor, verbose=-1),
        {**_BOOST_SLIDERS_TREE, "max_depth": {"type": "slider", "min": -1, "max": 15, "default": -1, "step": 1}},
    )
if HAS_CATBOOST:
    REGRESSION_MODELS["CatBoost"] = (partial(CatBoostRegressor, verbose=0), dict(_CATBOOST_SLIDERS))

CLUSTERING_MODELS = {
    "K-Means": (partial(KMeans, n_init=10, random_state=42), {
        "n_clusters": {"type": "slider", "min": 2, "max": 15, "default": 3, "step": 1},
    }),
    "DBSCAN": (DBSCAN, {
        "eps": {"type": "slider", "min": 0.1, "max": 5.0, "default": 0.5, "step": 0.1},
        "min_samples": {"type": "slider", "min": 2, "max": 20, "default": 5, "step": 1},
    }),
}

DIMRED_MODELS = {
    "PCA": (partial(PCA, random_state=42), {
        "n_components": {"type": "slider", "min": 2, "max": 10, "default": 2, "step": 1},
    }),
}

CATEGORIES = {
    "Classification": CLASSIFICATION_MODELS,
    "Regression": REGRESSION_MODELS,
    "Clustering": CLUSTERING_MODELS,
    "Dimensionality Reduction": DIMRED_MODELS,
}

# Candidate values per hyperparameter for automated GridSearch / RandomizedSearch.
# Keyed by algorithm display name (param names are shared across the
# classification/regression variants, so one grid serves both). Algorithms with
# no meaningful hyperparameters (Linear Regression, Naive Bayes) are absent and
# fall back to a plain fit.
SEARCH_SPACES = {
    "Logistic Regression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "Decision Tree": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
    "SVM": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]},
    "KNN": {"n_neighbors": [3, 5, 7, 11]},
    "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 6]},
    "LightGBM": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [-1, 5]},
    "CatBoost": {"iterations": [100, 200], "learning_rate": [0.05, 0.1], "depth": [4, 6]},
}


def count_grid_combos(grid: dict) -> int:
    total = 1
    for values in grid.values():
        total *= len(values)
    return total


def detect_column_types(df: pd.DataFrame) -> dict:
    return {
        col: "categorical" if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() < 15
        else "numeric"
        for col in df.columns
    }


def handle_missing_values(df: pd.DataFrame, strategy_per_column: dict) -> pd.DataFrame:
    df = df.copy()
    for col, strategy in strategy_per_column.items():
        if strategy == "drop":
            df = df.dropna(subset=[col])
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode_categoricals(df: pd.DataFrame, columns: list, method: str = "onehot") -> pd.DataFrame:
    df = df.copy()
    if not columns:
        return df
    if method == "onehot":
        df = pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == "label":
        for col in columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def _make_scaler(method: str):
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    return None


def scale_features(X_train, X_test, method: str):
    scaler = _make_scaler(method)
    if scaler is None:
        return X_train, X_test
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def scale_matrix(X, method: str):
    """Scale a single feature matrix (for unsupervised algorithms, no split)."""
    scaler = _make_scaler(method)
    if scaler is None:
        return X.values if hasattr(X, "values") else X
    return scaler.fit_transform(X)


def is_classification_target(y: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 15
