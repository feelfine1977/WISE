import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from wise.ui import state

# Optional SHAP import
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def render_refinement_page():
    st.header("4. Refinement – root-cause exploration")

    ds = state.get_dataset_state()
    results = state.get_results_state()

    if ds is None or results is None:
        st.info("Please complete steps 1–3 (Data, Norm, Results) before refinement.")
        return

    df_log = ds.df
    case_scores = results.case_scores  # ResultsState.case_scores
    slice_cols = ds.slice_cols

    # Case-level attributes per case
    case_attrs = df_log[[ds.case_id_col] + slice_cols].drop_duplicates()
    joined = case_scores.merge(case_attrs, on=ds.case_id_col, how="left")

    tab_global, tab_local, tab_outliers = st.tabs(
        ["Global dimension ranking", "Local explanation (SHAP)", "Outliers (IF/LOF)"]
    )

    # TAB 1
    with tab_global:
        _render_global_dimension_ranking(joined, ds, slice_cols)

    # TAB 2
    with tab_local:
        _render_local_shap_explanation(joined, ds, slice_cols)

    # TAB 3
    with tab_outliers:
        _render_outlier_detection(joined, ds, slice_cols)


# ===================================================================
# TAB 1: Global dimension ranking
# ===================================================================

def _render_global_dimension_ranking(joined: pd.DataFrame, ds, slice_cols: list[str]):
    st.subheader("Global dimension ranking")

    st.markdown(
        """
This view ranks dimensions (slice attributes) by how strongly they separate
good vs bad behaviour according to WISE.

You first select a target (score, badness, layer violation, constraint
violation), then dimensions are ranked by the maximum absolute deviation of
their category means from the global mean.
        """
    )

    # ----- Choose target metric -----
    target_options = {"Overall score (lower = worse)": "score"}
    target_options["Badness (1 - score) (higher = worse)"] = "1_minus_score"

    layer_cols = [c for c in joined.columns if c.startswith("violation_")]
    for col in layer_cols:
        lid = col.replace("violation_", "")
        label = f"Layer violation: {lid} (higher = worse)"
        target_options[label] = col

    constraint_cols = [c for c in joined.columns if c.startswith("viol_")]
    for col in constraint_cols:
        label = f"Constraint violation: {col[5:]} (higher = worse)"
        target_options[label] = col

    target_label = st.selectbox(
        "Target metric",
        list(target_options.keys()),
    )
    target_col = target_options[target_label]

    if target_col == "1_minus_score":
        joined["target"] = 1.0 - joined["score"]
    else:
        joined["target"] = joined[target_col].astype(float)

    target = joined["target"]
    global_mean = target.mean()

    st.markdown(
        f"Global mean of selected target: **{global_mean:.3f}** "
        "(higher typically means worse for badness/violations)."
    )

    # ----- Dimension ranking -----
    if not slice_cols:
        st.info("No slice dimensions defined; dimension-level refinement is not available.")
        return

    dims_to_consider = st.multiselect(
        "Dimensions to consider",
        slice_cols,
        default=slice_cols,
        help="These are the columns over which we will rank 'root-cause' potential.",
    )

    if not dims_to_consider:
        st.info("Select at least one dimension.")
        return

    dim_rows = []
    for dim in dims_to_consider:
        if dim not in joined.columns:
            continue
        g = joined.groupby(dim, dropna=False)["target"].mean()
        if g.empty:
            continue
        gaps = g - global_mean
        max_abs_gap = gaps.abs().max()
        n_categories = g.index.nunique()
        dim_rows.append(
            {
                "dimension": dim,
                "max_abs_gap": max_abs_gap,
                "n_categories": n_categories,
            }
        )

    if not dim_rows:
        st.info("No valid dimensions for refinement.")
        return

    dim_df = pd.DataFrame(dim_rows).sort_values("max_abs_gap", ascending=False)
    st.dataframe(dim_df)

    st.caption(
        """
`max_abs_gap` = largest absolute difference between a category mean and the
global mean. Dimensions with larger values tend to have categories that are
much better/worse than average and are therefore promising candidates for
root-cause investigation.
        """
    )

    # ----- Drill-down on one dimension -----
    st.subheader("Drill-down on a selected dimension")

    dim_selected = st.selectbox(
        "Select dimension for detailed view",
        dim_df["dimension"].tolist(),
    )

    g = joined.groupby(dim_selected, dropna=False)["target"].agg(["mean", "count"])
    g = g.rename(columns={"mean": "mean_target", "count": "n_cases"})
    g["gap"] = g["mean_target"] - global_mean
    g = g.sort_values("mean_target", ascending=False)

    st.markdown(
        f"Global mean target: **{global_mean:.3f}**. "
        f"Table sorted by mean target (higher = worse for badness/violations)."
    )
    st.dataframe(g.head(50))

    st.subheader(f"Mean target by {dim_selected}")
    top_n = st.number_input(
        "Number of categories to show",
        min_value=1,
        max_value=int(len(g)),
        value=min(20, len(g)),
        step=1,
    )

    g_plot = g.head(top_n).reset_index()
    fig_bar = px.bar(
        g_plot,
        x=dim_selected,
        y="mean_target",
        color="gap",
        color_continuous_scale="RdYlGn_r",
        title=f"Mean target by {dim_selected}",
        labels={"mean_target": "mean target", "gap": "gap vs global"},
    )
    fig_bar.update_layout(xaxis_title=dim_selected, yaxis_title="Mean target")
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown(
        """
Categories at the **right/top** with high bars and red colours are worse
than average for the selected target. Categories with low bars and green colours
are better than average.
        """
    )


# ===================================================================
# TAB 2: Local SHAP explanation
# ===================================================================

def _render_local_shap_explanation(joined: pd.DataFrame, ds, slice_cols: list[str]):
    st.subheader("Local explanation using SHAP")

    st.markdown(
        """
This view trains a simple model on case-level attributes to approximate the
selected target metric (score or violation) and then uses SHAP to explain the
prediction for individual cases and slices.
        """
    )

    if not HAS_SHAP:
        st.warning(
            "The 'shap' package is not installed. "
            "Install it with `pip install shap` in your environment to use this feature."
        )
        return

    # ----- Choose target metric -----
    target_options = {"Overall score (lower = worse)": "score"}
    target_options["Badness (1 - score) (higher = worse)"] = "1_minus_score"

    layer_cols = [c for c in joined.columns if c.startswith("violation_")]
    for col in layer_cols:
        lid = col.replace("violation_", "")
        label = f"Layer violation: {lid} (higher = worse)"
        target_options[label] = col

    constraint_cols = [c for c in joined.columns if c.startswith("viol_")]
    for col in constraint_cols:
        label = f"Constraint violation: {col[5:]} (higher = worse)"
        target_options[label] = col

    target_label = st.selectbox(
        "Target metric for SHAP",
        list(target_options.keys()),
        key="shap_target_select",
    )
    target_col = target_options[target_label]

    if target_col == "1_minus_score":
        joined["target"] = 1.0 - joined["score"]
    else:
        joined["target"] = joined[target_col].astype(float)

    y = joined["target"].values
    global_mean = float(np.mean(y))

    st.markdown(
        f"Global mean of selected target: **{global_mean:.3f}** "
        "(higher typically means worse for badness/violations)."
    )

    # ----- Feature selection -----
    st.markdown(
        """
Select which columns to use as explanatory features. These are usually the
slice dimensions and derived attributes (company, spend area, vendor, duration
bucket, month, ...).
        """
    )
    if not slice_cols:
        st.info("No slice dimensions defined; cannot build feature set.")
        return

    feature_cols = st.multiselect(
        "Columns to use as SHAP features",
        slice_cols,
        default=slice_cols,
        key="shap_feature_cols",
        help=(
            "These columns will be encoded and used as features for the SHAP model. "
            "Using too many high-cardinality columns may increase compute time."
        ),
    )
    if not feature_cols:
        st.info("Select at least one feature column.")
        return

    # ----- Sampling & model config -----
    st.markdown("### Sampling and model configuration")
    max_samples = st.number_input(
        "Maximum number of cases to use for training/explanation",
        min_value=100,
        max_value=int(len(joined)),
        value=min(2000, len(joined)),
        step=100,
        key="shap_max_samples",
    )
    random_state = st.number_input(
        "Random seed",
        min_value=0,
        max_value=10_000,
        value=42,
        step=1,
        key="shap_seed",
    )

    train_clicked = st.button("Train / update SHAP model")

    # Decide whether we need to (re-)train:
    shap_state = st.session_state.get("wise_shap_state")
    need_train = train_clicked or shap_state is None

    if need_train:
        if len(joined) > max_samples:
            joined_sample = joined.sample(n=int(max_samples), random_state=random_state)
        else:
            joined_sample = joined.copy()

        # --- Build feature matrix X ---
        X_raw = joined_sample[feature_cols].copy()

        # 1) Booleans → ints, otherwise pandas tends to keep an object-ish mix
        bool_cols = X_raw.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            X_raw[bool_cols] = X_raw[bool_cols].astype(int)

        # 2) One-hot encode categoricals (and keep NaNs as a category)
        X = pd.get_dummies(X_raw, dummy_na=True)

        # 3) SHAP's TreeExplainer wants pure float64, not a mixed/object array
        X = X.astype(float)

        # Target as float
        y_sample = joined_sample["target"].astype(float).values

        st.write(f"Training on {len(joined_sample)} cases with {X.shape[1]} encoded features.")

        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X, y_sample)

        # Use the generic API; SHAP will pick TreeExplainer for RandomForest
        explainer = shap.Explainer(model, X)

        # For tree models, it's often safer to disable the additivity check.
        # On older SHAP versions 'check_additivity' may not exist, so we try/except.
        try:
            shap_values = explainer(X, check_additivity=False)
        except TypeError:
            shap_values = explainer(X)

        sample_df = joined_sample[[ds.case_id_col] + feature_cols + ["target"]].copy()
        sample_df["__row_pos__"] = np.arange(len(sample_df))

        shap_state = {
            "X": X,
            "shap_values": shap_values,
            "sample_df": sample_df,
            "feature_cols": feature_cols,
            "target_label": target_label,
        }
        st.session_state["wise_shap_state"] = shap_state

        st.success("Model trained and SHAP values computed.")


    if shap_state is None:
        return

    _render_shap_case_and_slice(ds)


def _render_shap_case_and_slice(ds):
    shap_state = st.session_state.get("wise_shap_state")
    if shap_state is None:
        return

    X = shap_state["X"]
    shap_values = shap_state["shap_values"]
    sample_df = shap_state["sample_df"]
    feature_cols = shap_state["feature_cols"]

    # ---------- Case ranking table ----------
    st.subheader("Case ranking by target")
    sample_df_sorted = sample_df.sort_values("target", ascending=False)
    top_n_cases = st.number_input(
        "Number of worst cases to list",
        min_value=1,
        max_value=int(len(sample_df_sorted)),
        value=min(50, len(sample_df_sorted)),
        step=1,
        key="shap_top_cases",
    )
    st.dataframe(
        sample_df_sorted[[ds.case_id_col, "target"] + feature_cols].head(top_n_cases)
    )

    # ---------- Case-level explanation ----------
    st.subheader("Explain a specific case")

    worst_cases = sample_df_sorted[ds.case_id_col].head(top_n_cases).tolist()
    case_to_explain = st.selectbox(
        "Select a case to explain (sorted by worst target)",
        worst_cases,
    )

    row_info = sample_df_sorted[sample_df_sorted[ds.case_id_col] == case_to_explain][
        "__row_pos__"
    ]
    if row_info.empty:
        st.warning("Selected case not found in SHAP sample.")
        return
    row_pos = int(row_info.iloc[0])

    target_value = float(sample_df_sorted.loc[row_info.index[0], "target"])
    st.markdown(
        f"Explaining case **{case_to_explain}** (target={target_value:.3f})."
    )

    shap_row = shap_values.values[row_pos]
    feature_names = X.columns

    df_shap = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_row,
        }
    )
    df_shap["abs_shap"] = df_shap["shap_value"].abs()
    df_shap = df_shap.sort_values("abs_shap", ascending=False)

    top_k = st.number_input(
        "Number of top features to show (case explanation)",
        min_value=1,
        max_value=int(len(df_shap)),
        value=min(10, len(df_shap)),
        step=1,
        key="shap_top_features_case",
    )

    top = df_shap.head(top_k)

    st.markdown(
        """
Positive SHAP values push the prediction **up** (towards higher badness / violation),
negative SHAP values push it **down** (towards lower badness / closer to norm).
        """
    )

    fig = px.bar(
        top.sort_values("shap_value"),
        x="shap_value",
        y="feature",
        orientation="h",
        color="shap_value",
        color_continuous_scale="RdYlGn_r",
        title="Local SHAP explanation for selected case",
        labels={"shap_value": "SHAP value"},
    )
    fig.update_layout(yaxis_title="Feature", xaxis_title="Contribution to target")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Raw feature values for this case")
    st.json(
        sample_df_sorted.loc[row_info.index[0], feature_cols + ["target"]].to_dict()
    )

    # ---------- Slice-level SHAP ----------
    st.subheader("Slice-level SHAP (aggregate)")

    st.markdown(
        """
Instead of a single case, you can also look at a **slice** (e.g. a particular
day of week, duration bucket, or spend area). We aggregate SHAP values across
all sample cases in that slice and show the features that most systematically
push the target up or down for that slice.
        """
    )

    dims_for_slice = [
        c for c in feature_cols if c in sample_df.columns and c != "__row_pos__"
    ]
    if not dims_for_slice:
        st.info("No suitable dimensions for slice-level SHAP (features not found).")
        return

    dim_slice = st.selectbox(
        "Dimension for slice-level SHAP",
        dims_for_slice,
        key="shap_slice_dim",
    )

    categories = (
        sample_df[dim_slice].dropna().astype(str).value_counts().index.tolist()
    )
    if not categories:
        st.info("No categories found for selected dimension.")
        return

    cat_slice = st.selectbox(
        f"Slice category in '{dim_slice}'",
        categories,
        key="shap_slice_cat",
    )

    mask = sample_df[dim_slice].astype(str) == cat_slice
    idxs = sample_df.index[mask]
    if not len(idxs):
        st.info("No cases in this slice in the SHAP sample.")
        return

    row_positions = sample_df.loc[idxs, "__row_pos__"].astype(int).values
    shap_slice = shap_values.values[row_positions, :]
    mean_shap = shap_slice.mean(axis=0)

    df_slice_shap = pd.DataFrame(
        {
            "feature": X.columns,
            "shap_value": mean_shap,
        }
    )
    df_slice_shap["abs_shap"] = df_slice_shap["shap_value"].abs()
    df_slice_shap = df_slice_shap.sort_values("abs_shap", ascending=False)

    top_k_slice = st.number_input(
        "Number of top features to show (slice explanation)",
        min_value=1,
        max_value=int(len(df_slice_shap)),
        value=min(10, len(df_slice_shap)),
        step=1,
        key="shap_top_features_slice",
    )

    top_slice = df_slice_shap.head(top_k_slice)

    st.markdown(
        f"Slice **{dim_slice} = {cat_slice!r}** has {len(row_positions)} sample cases. "
        "Bars show average SHAP contributions over these cases."
    )

    fig2 = px.bar(
        top_slice.sort_values("shap_value"),
        x="shap_value",
        y="feature",
        orientation="h",
        color="shap_value",
        color_continuous_scale="RdYlGn_r",
        title=f"Slice-level SHAP for {dim_slice} = {cat_slice}",
        labels={"shap_value": "mean SHAP value in slice"},
    )
    fig2.update_layout(yaxis_title="Feature", xaxis_title="Average contribution to target")
    st.plotly_chart(fig2, width="stretch")


# ===================================================================
# TAB 3: Outlier detection (IF / LOF)
# ===================================================================

def _render_outlier_detection(joined: pd.DataFrame, ds, slice_cols: list[str]):
    st.subheader("Outliers – model-based suggestions")

    st.markdown(
        """
This view uses unsupervised outlier detection to find **unusual cases** and
**unusual slices** based on their feature combinations, not only on the WISE
score. It can surface interesting patterns that are rare but structurally
different from the bulk of the data.
        """
    )

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        st.warning(
            "scikit-learn is required for outlier detection. "
            "Install it with `pip install scikit-learn`."
        )
        return

    # ----- Choose target metric (for context only) -----
    target_options = {"Overall score (lower = worse)": "score"}
    target_options["Badness (1 - score) (higher = worse)"] = "1_minus_score"

    target_label = st.selectbox(
        "Target metric (context only)",
        list(target_options.keys()),
        key="outlier_target_select",
    )
    target_col = target_options[target_label]

    if target_col == "1_minus_score":
        joined["target_out"] = 1.0 - joined["score"]
    else:
        joined["target_out"] = joined[target_col].astype(float)

    # ----- Feature selection -----
    st.markdown(
        """
Select which columns to use as features for outlier detection. These are
usually the same slice dimensions and derived attributes you used elsewhere.
        """
    )
    if not slice_cols:
        st.info("No slice dimensions defined; cannot build feature set.")
        return

    feature_cols = st.multiselect(
        "Columns to use as outlier features",
        slice_cols,
        default=slice_cols,
        key="outlier_feature_cols",
        help=(
            "These columns will be encoded and used as features for the outlier model. "
            "Using too many high-cardinality columns may increase compute time."
        ),
    )
    if not feature_cols:
        st.info("Select at least one feature column.")
        return

    # ----- Sampling & model config -----
    st.markdown("### Sampling and model configuration")
    max_samples = st.number_input(
        "Maximum number of cases to use for outlier detection",
        min_value=100,
        max_value=int(len(joined)),
        value=min(5000, len(joined)),
        step=100,
        key="outlier_max_samples",
    )
    random_state = st.number_input(
        "Random seed (outliers)",
        min_value=0,
        max_value=10_000,
        value=42,
        step=1,
        key="outlier_seed",
    )

    algo = st.selectbox(
        "Outlier detection algorithm",
        ["IsolationForest", "LocalOutlierFactor"],
        help=(
            "IsolationForest works well for many feature types and supports scoring "
            "new samples. LocalOutlierFactor is more local but does not easily score new data."
        ),
    )

    run_clicked = st.button("Run / update outlier detection")

    out_state = st.session_state.get("wise_outlier_state")
    need_run = run_clicked or out_state is None

    if need_run:
        # ----- Prepare data -----
        if len(joined) > max_samples:
            joined_sample = joined.sample(n=int(max_samples), random_state=random_state)
        else:
            joined_sample = joined.copy()

        X_raw = joined_sample[feature_cols]
        X = pd.get_dummies(X_raw, dummy_na=True)

        # ----- Fit model -----
        if algo == "IsolationForest":
            model = IsolationForest(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
                contamination="auto",
            )
            model.fit(X)
            outlier_score = -model.score_samples(X)  # higher = more outlying
        else:
            lof = LocalOutlierFactor(
                n_neighbors=20,
                contamination="auto",
                novelty=False,
            )
            y_pred = lof.fit_predict(X)
            outlier_score = -lof.negative_outlier_factor_  # higher = more outlying

        joined_sample = joined_sample.copy()
        joined_sample["outlier_score"] = outlier_score

        # Normalize for display
        if np.nanmax(outlier_score) > np.nanmin(outlier_score):
            joined_sample["outlier_score_norm"] = (
                joined_sample["outlier_score"] - joined_sample["outlier_score"].min()
            ) / (joined_sample["outlier_score"].max() - joined_sample["outlier_score"].min())
        else:
            joined_sample["outlier_score_norm"] = 0.0

        out_state = {
            "df": joined_sample,
            "feature_cols": feature_cols,
            "target_label": target_label,
            "algo": algo,
        }
        st.session_state["wise_outlier_state"] = out_state

        st.success(
            f"Outlier detection finished on {len(joined_sample)} cases "
            f"using {algo}."
        )
    else:
        st.info(
            "Using previously computed outlier scores. "
            "If you change target, features or algorithm, press "
            "'Run / update outlier detection' again."
        )

    if out_state is None:
        return

    _render_outlier_views(ds, slice_cols, out_state)


def _render_outlier_views(ds, slice_cols: list[str], out_state: dict):
    df_out = out_state["df"]
    feature_cols = out_state["feature_cols"]

    # ---------- Case-level outliers ----------
    st.subheader("Top outlier cases")

    out_sorted = df_out.sort_values("outlier_score_norm", ascending=False)
    top_n_cases = st.number_input(
        "Number of top outlier cases to show",
        min_value=1,
        max_value=int(len(out_sorted)),
        value=min(50, len(out_sorted)),
        step=1,
        key="outlier_top_cases",
    )

    st.dataframe(
        out_sorted[
            [ds.case_id_col, "outlier_score_norm", "target_out"] + feature_cols
        ].head(top_n_cases)
    )

    st.caption(
        """
`outlier_score_norm` is a normalized outlier score (higher = more unusual).
`target_out` is the selected WISE-related target (score/badness/violation) shown
for context, so you can see which outliers are also bad according to the norm
and which are just unusual.
        """
    )

    fig_scatter = px.scatter(
        out_sorted,
        x="outlier_score_norm",
        y="target_out",
        hover_data=[ds.case_id_col] + feature_cols,
        title="Outlier score vs WISE target",
        labels={"outlier_score_norm": "outlier score (norm)", "target_out": "WISE target"},
    )
    st.plotly_chart(fig_scatter, width="stretch")

    # ---------- Slice-level outliers ----------
    st.subheader("Slice-level outliers by dimension")

    if not slice_cols:
        st.info("No slice dimensions defined.")
        return

    dim_slice = st.selectbox(
        "Dimension for slice-level outlier view",
        slice_cols,
        key="outlier_slice_dim",
    )

    # Clean aggregation: one column per metric, no MultiIndex problems
    g = df_out.groupby(dim_slice, dropna=False).agg(
        mean_outlier=("outlier_score_norm", "mean"),
        n_cases=("outlier_score_norm", "count"),
        mean_target=("target_out", "mean"),
    )
    g = g.sort_values("mean_outlier", ascending=False)

    st.dataframe(g.head(50))

    st.caption(
        """
`mean_outlier` = average normalized outlier score per category. Categories at
the top are slices that contain unusually many outlying cases, even if their
average WISE target is not the worst.
        """
    )

    top_n_slices = st.number_input(
        "Number of top slices to show",
        min_value=1,
        max_value=int(len(g)),
        value=min(20, len(g)),
        step=1,
        key="outlier_top_slices",
    )

    g_plot = g.head(top_n_slices).reset_index()
    fig_bar = px.bar(
        g_plot,
        x=dim_slice,
        y="mean_outlier",
        color="mean_target",
        color_continuous_scale="RdYlGn_r",
        title=f"Mean outlier score by {dim_slice}",
        labels={"mean_outlier": "mean outlier score", "mean_target": "mean WISE target"},
    )
    fig_bar.update_layout(xaxis_title=dim_slice, yaxis_title="Mean outlier score")
    st.plotly_chart(fig_bar, width="stretch")
