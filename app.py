import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score,
    silhouette_score,
)
from sklearn.decomposition import PCA

from ml_logic import (
    CLASSIFICATION_MODELS, REGRESSION_MODELS, CLUSTERING_MODELS, DIMRED_MODELS,
    SEARCH_SPACES, count_grid_combos,
    detect_column_types, handle_missing_values, encode_categoricals,
    scale_features, scale_matrix, is_classification_target,
)

st.set_page_config(page_title="Visual ML", page_icon=":material/analytics:", layout="wide")

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
/* Mode selector in the sidebar — spaced-out card-style options */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 10px;
    margin-top: 6px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 14px 16px;
    width: 100%;
    transition: border-color 0.15s ease, background 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(255,75,75,0.08);
    border-color: rgba(255,75,75,0.45);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(255,75,75,0.12);
    border-color: rgba(255,75,75,0.7);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    font-size: 1.02rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.title(":material/analytics: Visual ML")
st.caption("Upload data, configure preprocessing, pick an algorithm, tune it — no code required.")

uploaded = st.sidebar.file_uploader(":material/upload_file: Upload CSV", type=["csv"])
st.sidebar.caption("Your data never leaves this session.")

MODES = {
    "Supervised — predict a column": "supervised",
    "Clustering — group rows": "clustering",
    "Dimensionality Reduction": "dimred",
}
mode_label = st.sidebar.radio(":material/category: What do you want to do?", list(MODES.keys()))
mode = MODES[mode_label]

# Clear stale results when the user switches mode.
if st.session_state.get("active_mode") != mode:
    st.session_state["active_mode"] = mode
    st.session_state.pop("results", None)

if uploaded is None:
    st.info("Upload a CSV from the sidebar to get started.", icon=":material/upload:")
    st.stop()

df = pd.read_csv(uploaded)
col_types = detect_column_types(df)

tab_data, tab_prep, tab_model, tab_results = st.tabs(
    [":material/table_chart: Data", ":material/cleaning_services: Preprocess",
     ":material/settings: Model", ":material/monitoring: Results"]
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
    if mode == "supervised":
        st.subheader("Target & features")
        target_col = st.selectbox(":material/adjust: Target column (what to predict)", df.columns)
        feature_cols = [c for c in df.columns if c != target_col]
    else:
        st.subheader("Features")
        target_col = None
        feature_cols = list(df.columns)
    selected_features = st.multiselect(":material/view_column: Feature columns", feature_cols, default=feature_cols)

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
    if mode == "supervised":
        task = "classification" if is_classification_target(df[target_col]) else "regression"
        task_icon = ":material/category:" if task == "classification" else ":material/trending_up:"
        st.subheader(f"Detected task: {task_icon} {task.capitalize()}")
        model_registry = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    elif mode == "clustering":
        task = "clustering"
        st.subheader(":material/scatter_plot: Clustering")
        model_registry = CLUSTERING_MODELS
    else:
        task = "dimred"
        st.subheader(":material/compress: Dimensionality Reduction")
        model_registry = DIMRED_MODELS

    model_name = st.selectbox("Algorithm", list(model_registry.keys()))
    _, param_specs = model_registry[model_name]

    # Hyperparameters: either set manually, or searched automatically via CV.
    tuning_mode = "manual"
    cv_folds, n_iter = 3, 10
    search_grid = SEARCH_SPACES.get(model_name) if mode == "supervised" else None

    if mode == "supervised" and search_grid:
        st.subheader("Hyperparameter tuning")
        tuning_label = st.radio(
            "Tuning mode", ["Manual", "Grid Search CV", "Random Search CV"],
            horizontal=True,
            help="Manual: pick values yourself. Grid/Random Search: cross-validate to find the best combination automatically.",
        )
        tuning_mode = {"Manual": "manual", "Grid Search CV": "grid", "Random Search CV": "random"}[tuning_label]

    params = {}
    if tuning_mode == "manual":
        if param_specs:
            st.subheader("Hyperparameters")
            cols = st.columns(min(3, max(1, len(param_specs))))
            for i, (param, spec) in enumerate(param_specs.items()):
                with cols[i % len(cols)]:
                    if spec["type"] == "slider":
                        is_int = isinstance(spec["min"], int)
                        max_val = spec["max"]
                        # PCA can't extract more components than features available.
                        if model_name == "PCA" and param == "n_components":
                            max_val = max(2, min(spec["max"], len(selected_features)))
                        params[param] = st.slider(param, spec["min"], max_val, min(spec["default"], max_val), spec["step"])
                        if is_int:
                            params[param] = int(params[param])
                    elif spec["type"] == "select":
                        params[param] = st.selectbox(param, spec["options"], index=spec["options"].index(spec["default"]))
        elif mode == "supervised":
            st.caption("Auto-tuning isn't available here — this algorithm has no tunable hyperparameters.")
        else:
            st.caption("This algorithm has no tunable hyperparameters.")
    else:
        st.subheader("Search settings")
        combos = count_grid_combos(search_grid)
        cfg = st.columns(2)
        with cfg[0]:
            cv_folds = st.slider("CV folds", 2, 5, 3)
        if tuning_mode == "random":
            n_iter_max = min(30, combos)
            with cfg[1]:
                if n_iter_max > 2:
                    n_iter = st.slider("Search iterations", 2, n_iter_max, min(10, n_iter_max))
                else:
                    n_iter = combos
                    st.caption("Searching all combinations.")
            fits = n_iter * cv_folds
        else:
            fits = combos * cv_folds
        st.caption(f"Search space: **{combos}** combinations · ≈ **{fits}** model fits ({cv_folds}-fold CV).")
        if fits > 150:
            st.warning("That's a lot of fits — may be slow on limited hardware. Try Random Search or fewer folds.")
        with st.expander("Show search grid"):
            st.json(search_grid)

    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100 if mode == "supervised" else None

    if not selected_features:
        st.warning("Select at least one feature column in the Preprocess tab.")
    elif st.session_state.get("training"):
        # Second pass: button is locked while the model runs.
        verb = "Training" if mode == "supervised" else "Running"
        st.button(f":material/hourglass_top: {verb}…", type="primary", disabled=True, use_container_width=True)
        with st.spinner(f"{verb} the model…"):
            model_cls, _ = model_registry[model_name]

            if mode == "supervised":
                work_df = encode_categoricals(df[selected_features + [target_col]].copy(), cat_features, encoding_method)
                X = work_df.drop(columns=[target_col])
                y = work_df[target_col]
                # Boosting libs require integer-encoded class labels; encode any
                # non-numeric target (object/str/category dtypes).
                if task == "classification" and not pd.api.types.is_numeric_dtype(y):
                    y = y.astype("category").cat.codes

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                X_train, X_test = scale_features(X_train, X_test, scaling_method)

                result = {
                    "kind": task, "model_name": model_name,
                    "n_train": len(X_train), "n_test": len(X_test),
                }
                if tuning_mode == "manual":
                    model = model_cls(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    scoring = "accuracy" if task == "classification" else "r2"
                    base = model_cls()
                    if tuning_mode == "grid":
                        search = GridSearchCV(base, search_grid, cv=cv_folds, scoring=scoring, n_jobs=-1)
                    else:
                        search = RandomizedSearchCV(base, search_grid, n_iter=n_iter, cv=cv_folds,
                                                    scoring=scoring, n_jobs=-1, random_state=42)
                    search.fit(X_train, y_train)
                    y_pred = search.best_estimator_.predict(X_test)
                    lb = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "rank_test_score"]]
                    lb = lb.sort_values("rank_test_score").head(5).reset_index(drop=True)
                    lb["params"] = lb["params"].astype(str)
                    lb = lb.rename(columns={"mean_test_score": f"mean_cv_{scoring}", "rank_test_score": "rank"})
                    result["search"] = {
                        "method": "Grid Search" if tuning_mode == "grid" else "Random Search",
                        "best_params": search.best_params_,
                        "best_score": search.best_score_,
                        "scoring": scoring,
                        "leaderboard": lb,
                    }
                result["y_test"] = y_test
                result["y_pred"] = y_pred
                st.session_state["results"] = result
            else:
                work_df = encode_categoricals(df[selected_features].copy(), cat_features, encoding_method)
                X = scale_matrix(work_df, scaling_method)

                if mode == "clustering":
                    labels = model_cls(**params).fit_predict(X)
                    coords = PCA(n_components=2, random_state=42).fit_transform(X) if X.shape[1] > 2 else X[:, :2]
                    n_clusters = len(set(labels) - {-1})
                    sil = silhouette_score(X, labels) if 1 < n_clusters < len(X) else None
                    st.session_state["results"] = {
                        "kind": "clustering", "model_name": model_name,
                        "labels": labels, "coords": coords,
                        "n_clusters": n_clusters, "silhouette": sil,
                        "n_noise": int((labels == -1).sum()),
                    }
                else:  # dimred / PCA
                    n_comp = min(params.get("n_components", 2), X.shape[1])
                    pca = PCA(n_components=n_comp, random_state=42)
                    proj = pca.fit_transform(X)
                    st.session_state["results"] = {
                        "kind": "dimred", "model_name": model_name,
                        "projection": proj,
                        "explained_variance": pca.explained_variance_ratio_,
                    }

        st.session_state["training"] = False
        st.session_state["just_trained"] = True
        st.rerun()
    else:
        # First pass: lock the button and rerun so the disabled state shows.
        label = ":material/play_arrow: Train model" if mode == "supervised" else ":material/play_arrow: Run"
        if st.button(label, type="primary", use_container_width=True):
            st.session_state["training"] = True
            st.rerun()

def render_search_summary(res):
    """Show CV search results (best params + leaderboard) when present."""
    s = res.get("search")
    if not s:
        return
    st.markdown(f":material/search: **{s['method']} CV** — best `{s['scoring']}` = **{s['best_score']:.3f}**")
    st.write("Best hyperparameters:", s["best_params"])
    with st.expander("Top configurations (CV leaderboard)"):
        st.dataframe(s["leaderboard"], use_container_width=True)
    st.divider()


with tab_results:
    res = st.session_state.get("results")
    if res is None:
        st.info("Configure your model in the Model tab and run it.", icon=":material/tune:")
    elif res["kind"] == "classification":
        y_test, y_pred = res["y_test"], res["y_pred"]
        st.success(f"Trained **{res['model_name']}** on {res['n_train']:,} rows, tested on {res['n_test']:,} rows.")
        render_search_summary(res)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        c2.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.3f}")

        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Reds",
                        labels=dict(x="Predicted", y="Actual", color="Count"))
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    elif res["kind"] == "regression":
        y_test, y_pred = res["y_test"], res["y_pred"]
        st.success(f"Trained **{res['model_name']}** on {res['n_train']:,} rows, tested on {res['n_test']:,} rows.")
        render_search_summary(res)
        c1, c2 = st.columns(2)
        c1.metric("MSE", f"{mean_squared_error(y_test, y_pred):.3f}")
        c2.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", opacity=0.6, name="Predictions"))
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="red"), name="Ideal"))
        fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig, use_container_width=True)

    elif res["kind"] == "clustering":
        st.success(f"Ran **{res['model_name']}** — found **{res['n_clusters']}** cluster(s).")
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters", res["n_clusters"])
        c2.metric("Silhouette", f"{res['silhouette']:.3f}" if res["silhouette"] is not None else "—")
        c3.metric("Noise points", res["n_noise"])

        coords = res["coords"]
        fig = px.scatter(
            x=coords[:, 0], y=coords[:, 1],
            color=[str(l) for l in res["labels"]],
            labels={"x": "Component 1", "y": "Component 2", "color": "Cluster"},
            title="Clusters (projected to 2D)",
        )
        st.plotly_chart(fig, use_container_width=True)
        if res["silhouette"] is None:
            st.caption("Silhouette score needs 2+ clusters (excluding noise) to compute.")

    elif res["kind"] == "dimred":
        ev = res["explained_variance"]
        st.success(f"Ran **{res['model_name']}** — {len(ev)} components capture **{ev.sum()*100:.1f}%** of variance.")

        fig_var = px.bar(
            x=[f"PC{i+1}" for i in range(len(ev))], y=ev * 100,
            labels={"x": "Component", "y": "Explained variance (%)"},
            title="Explained Variance by Component",
        )
        st.plotly_chart(fig_var, use_container_width=True)

        proj = res["projection"]
        fig_proj = px.scatter(
            x=proj[:, 0], y=proj[:, 1],
            labels={"x": "PC1", "y": "PC2"},
            title="Data projected onto first 2 components",
        )
        st.plotly_chart(fig_proj, use_container_width=True)

# After a successful train, jump the user to the Results tab.
if st.session_state.pop("just_trained", False):
    components.html(
        """
        <script>
        const switchToResults = () => {
            const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
            for (const t of tabs) {
                if (t.innerText.includes("Results")) { t.click(); return; }
            }
        };
        setTimeout(switchToResults, 100);
        </script>
        """,
        height=0,
    )
