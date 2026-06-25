import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score

from ml_logic import (
    CLASSIFICATION_MODELS, REGRESSION_MODELS,
    detect_column_types, handle_missing_values, encode_categoricals,
    scale_features, is_classification_target,
)

st.set_page_config(page_title="Visual ML", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 2rem; }
h1 { font-weight: 800; }
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,75,75,0.12), rgba(255,75,75,0.02));
    border: 1px solid rgba(255,75,75,0.3);
    border-radius: 12px;
    padding: 14px 18px;
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    font-weight: 600;
}
div[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Visual ML")
st.caption("Upload data, configure preprocessing, pick an algorithm, tune it — no code required.")

uploaded = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])
st.sidebar.caption("Your data never leaves this session.")

if uploaded is None:
    st.info("👈 Upload a CSV from the sidebar to get started.")
    st.stop()

df = pd.read_csv(uploaded)
col_types = detect_column_types(df)

tab_data, tab_prep, tab_model, tab_results = st.tabs(
    ["📊 Data", "🧹 Preprocess", "⚙️ Model", "📈 Results"]
)

with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing cells", int(df.isna().sum().sum()))
    st.dataframe(df.head(20), use_container_width=True)

with tab_prep:
    st.subheader("Missing values")
    missing_cols = [c for c in df.columns if df[c].isna().any()]
    strategies = {}
    if missing_cols:
        cols = st.columns(min(3, len(missing_cols)))
        for i, col in enumerate(missing_cols):
            default = "drop" if col_types[col] == "categorical" else "mean"
            with cols[i % len(cols)]:
                strategies[col] = st.selectbox(
                    f"'{col}'", ["drop", "mean", "median", "mode"],
                    index=["drop", "mean", "median", "mode"].index(default),
                    key=f"missing_{col}",
                )
        df = handle_missing_values(df, strategies)
    else:
        st.success("No missing values detected.")

    st.divider()
    st.subheader("Target & features")
    target_col = st.selectbox("🎯 Target column (what to predict)", df.columns)
    feature_cols = [c for c in df.columns if c != target_col]
    selected_features = st.multiselect("✅ Feature columns", feature_cols, default=feature_cols)

    cat_features = [c for c in selected_features if col_types[c] == "categorical"]

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Encode categoricals")
        encoding_method = st.radio("Method", ["onehot", "label"], horizontal=True)
        if cat_features:
            st.caption(f"Detected: {', '.join(cat_features)}")
        else:
            st.caption("None among selected features.")
    with col_b:
        st.subheader("Scale numerics")
        scaling_method = st.radio("Method", ["none", "standard", "minmax"], horizontal=True)

with tab_model:
    task = "classification" if is_classification_target(df[target_col]) else "regression"
    st.subheader(f"Detected task: {'🔵 Classification' if task == 'classification' else '🟢 Regression'}")

    model_registry = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    model_name = st.selectbox("Algorithm", list(model_registry.keys()))
    _, param_specs = model_registry[model_name]

    if param_specs:
        st.subheader("Hyperparameters")
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
    else:
        params = {}
        st.caption("This algorithm has no tunable hyperparameters.")

    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
    train_clicked = st.button("🚀 Train model", type="primary", use_container_width=True)

with tab_results:
    if not train_clicked:
        st.info("Configure your model in the ⚙️ Model tab and click **Train model**.")
    else:
        with st.spinner("Training..."):
            work_df = df[selected_features + [target_col]].copy()
            work_df = encode_categoricals(work_df, cat_features, encoding_method)

            X = work_df.drop(columns=[target_col])
            y = work_df[target_col]

            if task == "classification" and y.dtype == "object":
                y = y.astype("category").cat.codes

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_train, X_test = scale_features(X_train, X_test, scaling_method)

            model_cls, _ = model_registry[model_name]
            model = model_cls(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        st.success(f"Trained **{model_name}** on {len(X_train):,} rows, tested on {len(X_test):,} rows.")

        if task == "classification":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("F1 Score", f"{f1:.3f}")

            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm, text_auto=True, color_continuous_scale="Reds",
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            fig.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            c1, c2 = st.columns(2)
            c1.metric("MSE", f"{mse:.3f}")
            c2.metric("R² Score", f"{r2:.3f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", opacity=0.6, name="Predictions"))
            lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="red"), name="Ideal"))
            fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
            st.plotly_chart(fig, use_container_width=True)
