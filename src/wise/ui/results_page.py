# src/wise/ui/results_page.py

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from wise.scoring.scoring import compute_case_scores
from wise.scoring.slices import aggregate_slices
from wise.norm import compute_view_weights
from wise.ui import state


def render_results_page():
    st.header("3. Run WISE and inspect results")

    ds = state.get_dataset_state()
    norm = state.get_norm_state()

    if norm is None:
        st.warning("Please load or create a norm on the 'Norm' page first.")
        return

    # --------------------------- Data source ---------------------------
    st.subheader("Data source")

    use_precomputed = st.checkbox(
        "Use precomputed scores (Parquet) instead of recomputing from raw log",
        value=False,
        help=(
            "Enable this if you have run precompute_scores.py and created a "
            "Parquet file with precomputed WISE scores for all views."
        ),
    )

    precomputed_df: Optional[pd.DataFrame] = None

    if use_precomputed:
        pre_file = st.text_input(
            "Precomputed scores file (Parquet)",
            "data/BPI_Challenge_2019_precomputed.parquet",
        )
        if st.button("Load precomputed"):
            try:
                precomputed_df = pd.read_parquet(pre_file)
                st.session_state["wise_precomputed_scores"] = precomputed_df
                st.success(f"Loaded precomputed scores from {pre_file}")
            except Exception as e:
                st.error(f"Could not load precomputed scores: {e}")
                return
        else:
            precomputed_df = st.session_state.get("wise_precomputed_scores")

        if precomputed_df is None:
            st.info("Load a precomputed Parquet file, or disable the checkbox to use the raw log.")
            return
    else:
        if ds is None:
            st.warning("Please upload data on the 'Data & Mapping' page first.")
            return

    # --------------------------- View & tuning ---------------------------
    view_names = norm.get_view_names()
    if not view_names:
        st.error("Norm contains no views.")
        return

    view_name = st.selectbox("View", view_names, index=0)

    st.subheader("Layer tuning (global what-if)")
    st.caption(
        "Each slider scales the importance of a whole layer within this view. "
        "`0` turns the layer off, `1` keeps the original importance, "
        "values above `1` make the layer more influential in the score."
    )

    # Discover which layers exist in the norm
    layer_ids = sorted({c.layer_id for c in norm.constraints})
    cols = st.columns(len(layer_ids)) if layer_ids else []

    layer_factors: Dict[str, float] = {}
    for idx, lid in enumerate(layer_ids):
        with cols[idx]:
            factor = st.slider(
                label=lid,
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help=(
                    "0: ignore this layer; 1: use the original view weight; "
                    ">1: boost this layer's impact on the score."
                ),
            )
        layer_factors[lid] = factor

    shrink_k = st.slider(
        "Shrinkage k (Empirical-Bayes for slice-level PI)",
        0.0,
        200.0,
        50.0,
        step=10.0,
        help=(
            "Empirical-Bayes shrinkage: small slices are pulled towards the "
            "global mean. Higher k = stronger pull for small slices. "
            "Set to 0 to disable."
        ),
    )

    if st.button("Compute scores and priorities"):
        _run_wise_and_show(
            ds=ds,
            norm=norm,
            view_name=view_name,
            layer_factors=layer_factors,
            shrink_k=shrink_k,
            precomputed_df=precomputed_df,
            use_precomputed=use_precomputed,
        )
    else:
        results_state = state.get_results_state()
        if results_state is not None:
            st.info(
                f"Showing previous results for view '{results_state.view_name}' "
                f"(k={results_state.params.get('shrink_k', 0)})."
            )
            _show_existing_results(results_state)
        else:
            st.info("Press 'Compute scores and priorities' to run WISE.")


# ----------------------------------------------------------------------
# Core scoring + slice aggregation
# ----------------------------------------------------------------------

def _build_weights_with_layer_factors(
    norm,
    view_name: str,
    layer_factors: Dict[str, float],
) -> Dict[str, float]:
    base = compute_view_weights(norm, view_name)
    scaled: Dict[str, float] = {}
    for c in norm.constraints:
        factor = layer_factors.get(c.layer_id, 1.0)
        scaled[c.id] = base[c.id] * factor

    total = sum(scaled.values())
    if total <= 0:
        n = len(scaled) or 1
        return {cid: 1.0 / n for cid in scaled}
    return {cid: w / total for cid, w in scaled.items()}


def _run_wise_and_show(
    ds,
    norm,
    view_name: str,
    layer_factors: Dict[str, float],
    shrink_k: float,
    precomputed_df: Optional[pd.DataFrame],
    use_precomputed: bool,
):
    # --- Case-level scores (raw or precomputed) ---
    if use_precomputed and precomputed_df is not None:
        st.info("Using precomputed scores.")
        if "view" in precomputed_df.columns:
            case_scores = precomputed_df[precomputed_df["view"] == view_name].copy()
        else:
            case_scores = precomputed_df.copy()
        # we assume 'score' and 'violation_<layer>' columns were precomputed
        df_log = None
    else:
        st.info("Computing scores from raw log.")
        df_log = ds.df
        weights_override = _build_weights_with_layer_factors(norm, view_name, layer_factors)
        case_scores = compute_case_scores(
            df=df_log,
            norm=norm,
            view_name=view_name,
            case_id_col=ds.case_id_col,
            activity_col=ds.activity_col,
            timestamp_col=ds.timestamp_col,
            weights_override=weights_override,
        )

    if "score" not in case_scores.columns:
        st.error("Case scores DataFrame has no 'score' column. Please adapt to your schema.")
        return

    st.subheader("Case scores (sample)")
    st.caption(
        "Each row is a case. Scores are in [0, 1]: higher = closer to the norm, "
        "lower = more deviating."
    )
    st.dataframe(case_scores.head(20))

    # --- Slice-level aggregation ---
    st.subheader("Slice-level priorities")

    if df_log is None and not use_precomputed:
        st.warning("Cannot aggregate slices: no base log available.")
        return

    if use_precomputed and df_log is None:
        # Heuristic: try to guess slice columns from precomputed file
        slice_cols = [c for c in case_scores.columns if c.startswith("case ") or c.startswith("case_")]
        case_id_col = next((c for c in case_scores.columns if "case" in c and "concept:name" in c), None)
        if case_id_col is None:
            st.warning("Could not identify case id column in precomputed file.")
            slice_summary = None
        else:
            # we can't call aggregate_slices without df_log; but we can emulate it
            slice_summary = _aggregate_slices_from_precomputed(
                case_scores,
                case_id_col=case_id_col,
                slice_cols=slice_cols,
                shrink_k=shrink_k,
            )
    else:
        slice_cols = ds.slice_cols
        if not slice_cols:
            st.info("No slice dimensions defined on the Data & Mapping page.")
            slice_summary = None
        else:
            slice_summary = aggregate_slices(
                df_scores=case_scores,
                df_log=df_log,
                case_id_col=ds.case_id_col,
                slice_cols=slice_cols,
                shrink_k=shrink_k,
            )

    state.set_results_state(
        view_name=view_name,
        case_scores=case_scores,
        slice_summary=slice_summary,
        params={"shrink_k": shrink_k, "use_precomputed": use_precomputed},
    )

    _show_existing_results(state.get_results_state())


def _aggregate_slices_from_precomputed(
    case_scores: pd.DataFrame,
    case_id_col: str,
    slice_cols: List[str],
    shrink_k: float,
) -> Optional[pd.DataFrame]:
    """Fallback aggregation when only precomputed scores are available."""
    if not slice_cols:
        return None

    df = case_scores[[case_id_col, "score"] + slice_cols].drop_duplicates()
    grouped = df.groupby(slice_cols, dropna=False)
    agg = grouped["score"].agg(["mean", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "mean_score", "count": "n_cases"})

    global_mean = float(df["score"].mean())
    agg["global_mean"] = global_mean
    agg["gap"] = global_mean - agg["mean_score"]

    if shrink_k > 0:
        agg["shrunk_mean_score"] = (
            (agg["n_cases"] * agg["mean_score"] + shrink_k * global_mean)
            / (agg["n_cases"] + shrink_k)
        )
    else:
        agg["shrunk_mean_score"] = agg["mean_score"]

    agg["PI"] = agg["n_cases"] * agg["gap"]
    return agg


# ----------------------------------------------------------------------
# Visualisation of existing results (case + slices + layer anomalies)
# ----------------------------------------------------------------------

def _show_existing_results(results_state):
    if results_state is None:
        st.info("No previous results available.")
        return

    ds = state.get_dataset_state()
    norm = state.get_norm_state()

    case_scores = results_state.case_scores
    slice_summary = results_state.slice_summary
    params = results_state.params or {}
    view_name = results_state.view_name

    st.subheader(f"Slice-level priorities for view '{view_name}'")
    if slice_summary is None or slice_summary.empty:
        st.info("No slice-level summary available.")
    else:
        st.dataframe(slice_summary.head(50))

        # Top slices by PI
        st.subheader("Top slices by Priority Index (PI)")
        slice_cols = [
            c for c in slice_summary.columns
            if c not in {"n_cases", "mean_score", "shrunk_mean_score", "gap", "PI", "global_mean"}
        ]
        if slice_cols:
            tmp = slice_summary.copy()
            tmp["slice_label"] = tmp[slice_cols].astype(str).agg(" | ".join, axis=1)
            chart_df = tmp.set_index("slice_label")[["PI"]].head(20)
            st.bar_chart(chart_df)
        else:
            st.info("No slice dimensions found in summary; skipping PI bar chart.")

    # ------------------------------------------------------------------ #
    # NEW: Layer × slice anomaly ranking & heatmap
    # ------------------------------------------------------------------ #
    st.subheader("Layer × slice anomaly ranking")

    if "score" not in case_scores.columns:
        st.info("Case scores missing; cannot compute layer anomalies.")
        return

    # Determine slice dimensions
    if slice_summary is not None and not slice_summary.empty:
        slice_cols_all = [
            c for c in slice_summary.columns
            if c not in {"n_cases", "mean_score", "shrunk_mean_score", "gap", "PI", "global_mean"}
        ]
    elif ds is not None:
        slice_cols_all = ds.slice_cols
    else:
        slice_cols_all = [c for c in case_scores.columns if c.startswith("case ") or c.startswith("case_")]

    if not slice_cols_all:
        st.info("No slice dimensions available; cannot compute layer anomalies.")
        return

    mode = st.radio(
        "Slice key mode",
        ["All slice dimensions (full key)", "Single dimension", "Custom combination (select 1–N dimensions)"],
        index=2,
        help=(
            "Full key: every combination of all slice dimensions.\n"
            "Single: one dimension at a time (e.g. spend area only).\n"
            "Custom: choose 1–N dimensions to build the slice key."
        ),
    )

    if mode == "All slice dimensions (full key)":
        key_cols = slice_cols_all
    elif mode == "Single dimension":
        dim_single = st.selectbox(
            "Dimension for heatmap",
            slice_cols_all,
            help="The slice key will be this dimension only.",
        )
        key_cols = [dim_single]
    else:  # custom combination
        key_cols = st.multiselect(
            "Dimensions for custom slice key",
            slice_cols_all,
            default=slice_cols_all[:2],
            help="Select 1–N dimensions to build the slice key.",
        )
        if not key_cols:
            st.info("Select at least one dimension for the custom slice key.")
            return

    # Layers to include
    layer_cols_map = {
        col[len("violation_") :]: col
        for col in case_scores.columns
        if col.startswith("violation_")
    }
    if not layer_cols_map:
        st.info("No layer violation columns (violation_*) found in case scores.")
        return

    layer_choices = sorted(layer_cols_map.keys())
    selected_layers = st.multiselect(
        "Layers to include",
        layer_choices,
        default=layer_choices,
        help="Only these layers will be considered in the anomaly ranking.",
    )
    if not selected_layers:
        st.info("Select at least one layer.")
        return

    # Volume shrinkage for layer priorities
    k_layer = st.number_input(
        "Volume shrinkage k for layer priorities",
        min_value=0.0,
        value=50.0,
        step=10.0,
        help=(
            "Controls how strongly small slices are down-weighted when ranking "
            "layer × slice anomalies. Set to 0 to ignore volume."
        ),
    )

    # Minimum cases per slice
    min_cases = st.number_input(
        "Min cases per slice",
        min_value=1,
        value=20,
        step=5,
    )

    # Number of cells (layer × slice) to consider
    top_n_cells = st.number_input(
        "Number of layer × slice cells to consider",
        min_value=1,
        value=12,
        step=1,
    )

    # Build merged data with slice attributes
    if ds is not None:
        df_log = ds.df
        key_df = df_log[[ds.case_id_col] + key_cols].drop_duplicates()
        merged = case_scores.merge(key_df, on=ds.case_id_col, how="left")
        case_id_col = ds.case_id_col
    else:
        # precomputed-only scenario: assume slice columns are already in case_scores
        merged = case_scores.copy()
        case_id_col = next((c for c in merged.columns if "case" in c and "concept:name" in c), None)
        if case_id_col is None:
            st.error("Could not determine case identifier column for anomaly computation.")
            return

    # Global stats per selected layer
    global_means: Dict[str, float] = {}
    global_stds: Dict[str, float] = {}
    for lid in selected_layers:
        col = layer_cols_map[lid]
        vals = merged[col].to_numpy()
        global_means[lid] = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        if not np.isfinite(std) or std <= 0:
            std = 0.0
        global_stds[lid] = std

    # Group by slice key and compute mean violations per layer + counts
    grouped = merged.groupby(key_cols, dropna=False)
    means = grouped[[layer_cols_map[lid] for lid in selected_layers]].mean()
    counts = grouped[case_id_col].size()

    if means.empty:
        st.info("No data for layer anomalies with the chosen key.")
        return

    # Build long-form table of layer × slice scores
    records = []
    for idx, idx_label in enumerate(means.index):
        # idx_label is scalar or tuple depending on number of key_cols
        if isinstance(idx_label, tuple):
            slice_values = list(idx_label)
        else:
            slice_values = [idx_label]
        slice_dict = {col_name: val for col_name, val in zip(key_cols, slice_values)}
        n_cases = int(counts.to_numpy()[idx])

        if n_cases < min_cases:
            continue

        for lid in selected_layers:
            col = layer_cols_map[lid]
            mean_v = float(means[col].to_numpy()[idx])
            global_mean = global_means[lid]
            std = global_stds[lid]
            gap = mean_v - global_mean
            if std > 0:
                z = gap / std
            else:
                z = 0.0
            z_plus = max(z, 0.0)
            if k_layer > 0:
                vol_weight = n_cases / (n_cases + k_layer)
            else:
                vol_weight = 1.0
            priority = z_plus * vol_weight
            pi = max(gap, 0.0) * n_cases

            rec = dict(slice_dict)
            rec.update(
                {
                    "slice_label": " | ".join(str(v) for v in slice_values),
                    "layer_id": lid,
                    "n_cases": n_cases,
                    "mean_violation": mean_v,
                    "global_mean": global_mean,
                    "gap": gap,
                    "z": z,
                    "z_plus": z_plus,
                    "vol_weight": vol_weight,
                    "priority": priority,
                    "PI": pi,
                }
            )
            records.append(rec)

    matrix = pd.DataFrame.from_records(records)
    if matrix.empty:
        st.info("No layer × slice cells passed the min-cases filter.")
        return

    matrix["PI_abs"] = matrix["PI"].abs()

    # Ranking metric
    rank_metric = st.selectbox(
        "Rank cells by",
        ["priority", "z_plus", "PI_abs", "gap", "PI"],
        index=0,
        help=(
            "priority = z_plus × volume weight (recommended)\n"
            "z_plus = severity (standardised gap, positive part)\n"
            "PI_abs = |n × gap|, classic priority index style\n"
            "gap = mean_violation - global_mean\n"
            "PI = n × gap"
        ),
    )
    ascending = False  # higher = worse for all these metrics

    ranked = matrix.sort_values(rank_metric, ascending=ascending)

    # Optional restriction to specific slices (by label)
    all_labels = ranked["slice_label"].unique().tolist()
    default_labels = ranked["slice_label"].head(top_n_cells).tolist()
    selected_labels = st.multiselect(
        "Optional: restrict to specific slices",
        options=all_labels,
        default=default_labels,
        help="Use this to focus on a subset of slices (e.g. specific spend areas or vendors).",
    )
    if selected_labels:
        ranked = ranked[ranked["slice_label"].isin(selected_labels)]

    top_df = ranked.head(top_n_cells).copy()

    st.subheader("Layer × slice priority table")
    display_cols = [
        "layer_id",
        "slice_label",
        "n_cases",
        "z_plus",
        "priority",
        "gap",
        "PI",
    ]
    st.dataframe(top_df[display_cols])

    # Heatmap metric
    heat_metric = st.selectbox(
        "Heatmap colour metric",
        ["gap", "z_plus", "priority"],
        index=0,
        help=(
            "gap = mean violation difference vs global.\n"
            "z_plus = standardised gap (positive part).\n"
            "priority = z_plus × volume weight."
        ),
    )

    heat_df = top_df.pivot(index="layer_id", columns="slice_label", values=heat_metric)
    fig = px.imshow(
        heat_df,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        labels={"color": f"{heat_metric} (layer deviation)"},
    )
    fig.update_layout(
        xaxis_title="Slice",
        yaxis_title="Layer",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
Cells show the average **deviation from global** for each (layer, slice)
combination, as measured by the chosen metric. Green ≈ better than global;
red ≈ worse than global. Use the priority table above to decide which
combinations deserve further attention given their severity and volume.
        """
    )

    # ------------------------------------------------------------------ #
    # BOX PLOT & SCORES HEATMAP
    # ------------------------------------------------------------------ #
    st.subheader("Boxplot by dimension")
    st.markdown(
        """
Select a dimension to see the distribution of case scores per category.
This is useful to understand how much variation there is inside a category
and how many outliers it contains.
        """
    )

    # reuse slice_cols_all from above if available, otherwise recompute
    if slice_summary is not None and not slice_summary.empty:
        slice_cols_all = [
            c for c in slice_summary.columns
            if c not in {"n_cases", "mean_score", "shrunk_mean_score", "gap", "PI", "global_mean"}
        ]
    elif ds is not None:
        slice_cols_all = ds.slice_cols
    else:
        slice_cols_all = [c for c in case_scores.columns if c.startswith("case ") or c.startswith("case_")]

    if slice_cols_all and ds is not None:
        dim_box = st.selectbox(
            "Dimension for boxplot",
            slice_cols_all,
            key="box_dim",
            help="For example: spend area, vendor, or company.",
        )
        joined = case_scores.merge(
            ds.df[[ds.case_id_col, dim_box]].drop_duplicates(),
            on=ds.case_id_col,
            how="left",
        )
        fig_box = px.box(
            joined,
            x=dim_box,
            y="score",
            points="outliers",
            color=dim_box,
        )
        fig_box.update_layout(
            xaxis_title=dim_box,
            yaxis_title="WISE score",
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No slice dimensions available for boxplot (or dataset not present).")

    st.subheader("Scores heatmap by dimension")
    st.markdown(
        """
Rows = categories of a chosen dimension (e.g. spend areas, vendors).
Columns = overall WISE score and layer-specific scores converted to a
score-like scale (1 = fully in norm, 0 = worst).
Colours = average **score** per category with green = better and red = worse.
        """
    )

    if slice_cols_all and ds is not None:
        dim_scores = st.selectbox(
            "Dimension for scores heatmap",
            slice_cols_all,
            key="scores_dim",
            help="For example: spend area or vendor.",
        )

        joined_scores = case_scores.merge(
            ds.df[[ds.case_id_col, dim_scores]].drop_duplicates(),
            on=ds.case_id_col,
            how="left",
        )

        metrics = ["score"]
        layer_violation_cols = [
            col for col in joined_scores.columns if col.startswith("violation_")
        ]
        for col in layer_violation_cols:
            score_col = col.replace("violation_", "score_")
            joined_scores[score_col] = 1.0 - joined_scores[col]
            metrics.append(score_col)

        group_s = joined_scores.groupby(dim_scores, dropna=False)
        score_matrix = group_s[metrics].mean().rename(columns={"score": "mean_score"})

        fig_scores = px.imshow(
            score_matrix,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            labels={"color": "mean score"},
        )
        fig_scores.update_layout(
            height=500,
            xaxis_title="Scores",
            yaxis_title=dim_scores,
        )
        st.plotly_chart(fig_scores, use_container_width=True)
    else:
        st.info("No slice dimensions available for scores heatmap (or dataset not present).")
