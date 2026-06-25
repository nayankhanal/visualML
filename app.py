import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from ml_logic import (
    CLASSIFICATION_MODELS, REGRESSION_MODELS,
    detect_column_types, handle_missing_values, encode_categoricals,
    scale_features, is_classification_target,
)

st.set_page_config(page_title="Visual ML", layout="wide")
st.title("Visual ML — pick an algorithm, tune it, see results")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV from the sidebar to get started.")
    st.stop()

df = pd.read_csv(uploaded)

st.subheader("1. Preview")
st.dataframe(df.head())

col_types = detect_column_types(df)

st.subheader("2. Missing values")
missing_cols = [c for c in df.columns if df[c].isna().any()]
strategies = {}
if missing_cols:
    for col in missing_cols:
        default = "drop" if col_types[col] == "categorical" else "mean"
        strategies[col] = st.selectbox(
            f"'{col}' has missing values — strategy",
            ["drop", "mean", "median", "mode"],
            index=["drop", "mean", "median", "mode"].index(default),
            key=f"missing_{col}",
        )
    df = handle_missing_values(df, strategies)
else:
    st.write("No missing values detected.")

st.subheader("3. Target column")
target_col = st.selectbox("Select the target (what you want to predict)", df.columns)

feature_cols = [c for c in df.columns if c != target_col]
st.subheader("4. Feature columns")
selected_features = st.multiselect("Select features to use", feature_cols, default=feature_cols)

cat_features = [c for c in selected_features if col_types[c] == "categorical"]
st.subheader("5. Encode categorical features")
encoding_method = st.radio("Encoding method", ["onehot", "label"], horizontal=True)
if cat_features:
    st.write(f"Categorical columns detected: {', '.join(cat_features)}")
else:
    st.write("No categorical features among your selection.")

st.subheader("6. Scale numeric features")
scaling_method = st.radio("Scaling method", ["none", "standard", "minmax"], horizontal=True)

st.subheader("7. Choose algorithm")
task = "classification" if is_classification_target(df[target_col]) else "regression"
st.write(f"Detected task type: **{task}**")

model_registry = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
model_name = st.selectbox("Algorithm", list(model_registry.keys()))
_, param_specs = model_registry[model_name]

st.subheader("8. Hyperparameters")
params = {}
cols = st.columns(min(3, max(1, len(param_specs))))
for i, (param, spec) in enumerate(param_specs.items()):
    with cols[i % len(cols)]:
        if spec["type"] == "slider":
            is_int = isinstance(spec["min"], int)
            params[param] = st.slider(
                param, spec["min"], spec["max"], spec["default"], spec["step"]
            )
            if is_int:
                params[param] = int(params[param])
        elif spec["type"] == "select":
            params[param] = st.selectbox(param, spec["options"], index=spec["options"].index(spec["default"]))

test_size = st.slider("Test set size (%)", 10, 50, 20) / 100

if st.button("Train model", type="primary"):
    work_df = df[selected_features + [target_col]].copy()
    work_df = encode_categoricals(work_df, cat_features, encoding_method)

    X = work_df.drop(columns=[target_col])
    y = work_df[target_col]

    if task == "classification" and y.dtype == "object":
        y = y.astype("category").cat.codes

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_test = scale_features(X_train, X_test, scaling_method)

    model_cls, _ = model_registry[model_name]
    model = model_cls(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("9. Results")
    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("F1 Score", f"{f1:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        c1, c2 = st.columns(2)
        c1.metric("MSE", f"{mse:.3f}")
        c2.metric("R² Score", f"{r2:.3f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)
