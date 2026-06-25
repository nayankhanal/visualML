import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

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
}

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
}


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


def scale_features(X_train, X_test, method: str):
    if method == "none":
        return X_train, X_test
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def is_classification_target(y: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 15
