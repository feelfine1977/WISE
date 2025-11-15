from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from wise.scoring.scoring import compute_case_scores
from wise.scoring.slices import (
    aggregate_slices,
    compute_slice_layer_matrix,
    compute_slice_constraint_matrix,
    rank_slices,
)
from wise.norm import compute_view_weights
from wise.ui import state


def render_results_page():
    st.header("3. Run WISE and inspect results")

    st.markdown(
        """
Use this page to:

1. Choose a **view** (e.g. Finance, Logistics). The view defines how each
   constraint in the norm is weighted.
2. Optionally adjust the **layer sliders**:
   - `1.0` = use the importance from the norm as-is,
   - `0.0` = ignore this layer,
   - `> 1.0` = emphasise this layer in the score.
3. Choose a **shrinkage k**. Higher values pull small slices more strongly
   towards the global average so rankings are less dominated by tiny, noisy
   segments.
4. Press **Compute scores and priorities** to recompute scores and rankings.
        """
    )

    ds = state.get_dataset_state()
    norm = state.get_norm_state()

    if ds is None:
        st.warning("Please upload data on the 'Data & Mapping' page first.")
        return
    if norm is None:
        st.warning("Please load or create a norm on the 'Norm' page first.")
        return

    view_names = norm.get_view_names()
    if not view_names:
        st.error("Norm contains no views.")
        return

    view_name = st.selectbox("View", view_names, index=0)

    # --- Layer boosters / toggles -------------------------------------------
    st.subheader("Layer tuning (global what-if)")
    st.caption(
        "Each slider scales the importance of a whole layer within this view. "
        "`0` turns the layer off, `1` keeps the original importance, "
        "values above `1` make the layer more influential in the score."
    )

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
        "Shrinkage k",
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
        )
    else:
        results = state.get_results_state()
        if results is not None:
            st.info(
                f"Showing results for view '{results.view_name}' "
                f"(k={results.params.get('shrink_k', 0)})."
            )
            _show_existing_results(results, ds, norm)
        else:
            st.info("Press **Compute scores and priorities** to run WISE.")


def _build_weights_with_layer_factors(
    norm,
    view_name: str,
    layer_factors: Dict[str, float],
) -> Dict[str, float]:
    """
    Take view-based weights, apply layer-level factors, and renormalise.
    """
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


def _run_wise_and_show(ds, norm, view_name: str, layer_factors: Dict[str, float], shrink_k: float):
    df = ds.df

    weights_override = _build_weights_with_layer_factors(norm, view_name, layer_factors)

    with st.spinner("Computing case-level scores..."):
        case_scores = compute_case_scores(
            df=df,
            norm=norm,
            view_name=view_name,
            case_id_col=ds.case_id_col,
            activity_col=ds.activity_col,
            timestamp_col=ds.timestamp_col,
            weights_override=weights_override,
        )

    st.subheader("Case scores (sample)")
    st.caption(
        "Each row is a case (e.g., one PO item). "
        "Scores are in [0, 1]: higher = closer to the norm, lower = more deviating."
    )
    st.dataframe(case_scores.head(20))

    if not ds.slice_cols:
        st.info("No slice dimensions defined; only case scores are available.")
        state.set_results_state(
            view_name, case_scores, None, params={"shrink_k": shrink_k}
        )
        return

    with st.spinner("Aggregating slices..."):
        slice_summary = aggregate_slices(
            df_scores=case_scores,
            df_log=df,
            case_id_col=ds.case_id_col,
            slice_cols=ds.slice_cols,
            shrink_k=shrink_k,
        )

    state.set_results_state(
        view_name=view_name,
        case_scores=case_scores,
        slice_summary=slice_summary,
        params={"shrink_k": shrink_k},
    )

    _show_existing_results(state.get_results_state(), ds, norm, case_scores=case_scores)


def _show_existing_results(results_state, ds, norm, case_scores=None):
    st.subheader("Slice-level priorities")
    st.markdown(
        """
**How to read this table**

- `n_cases`: number of cases in the slice.
- `mean_score`: average WISE score (1 = fully in norm, lower = worse).
- `shrunk_mean_score`: mean after shrinkage; small slices are pulled towards
  the global average to reduce noise.
- `gap`: `global_mean - slice_mean`. Positive gap means the slice is **worse**
  than the overall average; negative gap means **better**.
- `PI`: `n_cases × gap`. Higher positive values highlight slices that are
  both frequent and below the norm and are therefore promising candidates
  for further investigation.
        """
    )
    st.dataframe(results_state.slice_summary.head(50))

    slice_cols = ds.slice_cols

    # Bar chart of PI
    st.subheader("Top slices by PI")
    if slice_cols and results_state.slice_summary is not None and not results_state.slice_summary.empty:
        tmp = results_state.slice_summary.copy()
        tmp["slice_label"] = tmp[slice_cols].astype(str).agg(" | ".join, axis=1)
        chart_df = tmp.set_index("slice_label")[["PI"]].head(20)
        st.caption(
            "Bars to the right (positive PI) indicate slices that are **worse than the norm**. "
            "Bars to the left (negative PI) perform **better than average**."
        )
        st.bar_chart(chart_df)
    else:
        st.info("No slice information available for bar chart.")

    # Ensure we have case_scores
    if case_scores is None:
        case_scores = results_state.case_scores

    # ------------------------------------------------------------------ #
    # LAYER-LEVEL HEATMAP
    # ------------------------------------------------------------------ #
    st.subheader("Layer × slice heatmap")
    st.markdown(
        """
Rows are norm layers; columns are slices (combinations of the selected slice
attributes or a single dimension, depending on the mode).  

Colours show how much the **average violation** in that slice differs from the
global average for that layer:

- green ≈ lower violation than global (better),
- red ≈ higher violation than global (worse),
- values near 0 ≈ similar to global.
        """
    )

    if not slice_cols:
        st.info("No slice dimensions defined; cannot build heatmap.")
        return

    mode = st.radio(
        "Heatmap mode",
        ["All slice dimensions (full key)", "Single dimension"],
        index=0,
        horizontal=True,
    )

    if mode == "All slice dimensions (full key)":
        key_cols = slice_cols
        ranking_source = results_state.slice_summary
        slice_layer = compute_slice_layer_matrix(
            df_scores=case_scores,
            df_log=ds.df,
            case_id_col=ds.case_id_col,
            slice_cols=key_cols,
        )
    else:
        dim = st.selectbox(
            "Dimension for heatmap",
            slice_cols,
            help="Pick one attribute, e.g. vendor or spend area, to compare its categories.",
        )
        key_cols = [dim]
        shrink_k_dim = results_state.params.get("shrink_k", 0.0)
        ranking_source = aggregate_slices(
            df_scores=case_scores,
            df_log=ds.df,
            case_id_col=ds.case_id_col,
            slice_cols=key_cols,
            shrink_k=shrink_k_dim,
        )
        slice_layer = compute_slice_layer_matrix(
            df_scores=case_scores,
            df_log=ds.df,
            case_id_col=ds.case_id_col,
            slice_cols=key_cols,
        )

    if slice_layer.empty or ranking_source is None or ranking_source.empty:
        st.info("No slice-layer data available for heatmap.")
        return

    st.markdown("**Reprioritisation controls**")
    col1, col2, col3 = st.columns(3)
    with col1:
        metric = st.selectbox(
            "Rank slices by",
            ["PI", "PI_abs", "gap", "gap_abs"],
            index=0,
            help=(
                "`PI`: raw priority index; `PI_abs`: by absolute PI (large positive or negative), "
                "`gap`: by gap alone; `gap_abs`: by absolute gap."
            ),
        )
    with col2:
        min_cases = st.number_input(
            "Min cases per slice",
            min_value=1,
            value=50,
            step=10,
            help="Ignore tiny slices by requiring at least this many cases per slice.",
        )
    with col3:
        top_n = st.number_input(
            "Number of slices to consider",
            min_value=1,
            value=10,
            step=1,
            help="Only consider the top N slices according to the ranking metric above.",
        )

    ranked = rank_slices(
        ranking_source,
        min_cases=min_cases,
        top_n=top_n,
        metric=metric,
    )
    if ranked.empty:
        st.info("No slices satisfy the filter; try lowering 'Min cases per slice'.")
        return

    merged = ranked.merge(slice_layer, on=key_cols, how="left")
    merged["slice_label"] = merged[key_cols].astype(str).agg(" | ".join, axis=1)

    available_labels = merged["slice_label"].tolist()
    selected_labels = st.multiselect(
        "Optional: restrict to specific slices",
        options=available_labels,
        default=available_labels,
        help="Use this if you want to focus on selected vendors/companies/categories "
             "instead of all top slices.",
    )
    if selected_labels:
        merged = merged[merged["slice_label"].isin(selected_labels)]
    if merged.empty:
        st.info("No slices left after manual selection.")
        return

    layer_gap_cols = [
        c for c in merged.columns
        if c.startswith("violation_") and c.endswith("_gap")
    ]
    if not layer_gap_cols:
        st.info("No layer gap columns found in slice-layer matrix.")
        return

    data = {}
    for col in layer_gap_cols:
        layer_id = col.replace("violation_", "").replace("_gap", "")
        data[layer_id] = merged[col].values

    heatmap_df = pd.DataFrame(data, index=merged["slice_label"]).T  # layers × slices

    fig = px.imshow(
        heatmap_df,
        color_continuous_scale="RdYlGn_r",  # green = better, red = worse
        aspect="auto",
        labels={"color": "gap (violation - global)"},
    )
    fig.update_layout(height=400, xaxis_title="Slice", yaxis_title="Layer")
    st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------------ #
    # CONSTRAINT-LEVEL HEATMAP (within a layer)
    # ------------------------------------------------------------------ #
    st.subheader("Constraint × slice heatmap (within a layer)")
    st.markdown(
        """
Use this view to drill down inside a specific layer.  
For example: select `presence` to see each presence constraint (GR, INV, CLEAR, …)
for the same slices, or select `exclusion` or `order_lag` to see which concrete
rules are most problematic.
        """
    )

    dim_c = st.selectbox(
        "Dimension for constraint heatmap",
        slice_cols,
        help="For example: vendor or spend area.",
    )

    layer_ids_all = sorted({c.layer_id for c in norm.constraints})
    layer_for_constraints = st.selectbox(
        "Layer",
        layer_ids_all,
        help="Only constraints from this layer will appear as rows in the heatmap.",
    )

    shrink_k_c = results_state.params.get("shrink_k", 0.0)
    ranking_source_c = aggregate_slices(
        df_scores=case_scores,
        df_log=ds.df,
        case_id_col=ds.case_id_col,
        slice_cols=[dim_c],
        shrink_k=shrink_k_c,
    )
    matrix_c = compute_slice_constraint_matrix(
        df_scores=case_scores,
        df_log=ds.df,
        case_id_col=ds.case_id_col,
        slice_cols=[dim_c],
        norm=norm,
        layer_id=layer_for_constraints,
    )

    if matrix_c.empty or ranking_source_c.empty:
        st.info("No constraint-level data available for this layer/dimension.")
        return

    st.markdown("**Reprioritisation controls for constraint view**")
    col1c, col2c, col3c = st.columns(3)
    with col1c:
        metric_c = st.selectbox(
            "Rank slices by (constraint view)",
            ["PI", "PI_abs", "gap", "gap_abs"],
            index=0,
        )
    with col2c:
        min_cases_c = st.number_input(
            "Min cases per slice (constraint view)",
            min_value=1,
            value=50,
            step=10,
        )
    with col3c:
        top_n_c = st.number_input(
            "Number of slices to consider (constraint view)",
            min_value=1,
            value=10,
            step=1,
        )

    ranked_c = rank_slices(
        ranking_source_c,
        min_cases=min_cases_c,
        top_n=top_n_c,
        metric=metric_c,
    )
    if ranked_c.empty:
        st.info("No slices satisfy the filter in constraint view.")
        return

    merged_c = ranked_c.merge(matrix_c, on=[dim_c], how="left")
    merged_c["slice_label"] = merged_c[dim_c].astype(str)

    labels_c = merged_c["slice_label"].tolist()
    selected_labels_c = st.multiselect(
        "Optional: restrict to specific slices (constraint view)",
        options=labels_c,
        default=labels_c,
    )
    if selected_labels_c:
        merged_c = merged_c[merged_c["slice_label"].isin(selected_labels_c)]
    if merged_c.empty:
        st.info("No slices left after selection in constraint view.")
        return

    constraint_gap_cols = [
        c for c in merged_c.columns
        if c.endswith("_gap") and not c.startswith("violation_")
    ]
    if not constraint_gap_cols:
        st.info("No constraint gap columns found.")
        return

    data_c = {}
    for col in constraint_gap_cols:
        cid = col.replace("_gap", "")
        data_c[cid] = merged_c[col].values

    heatmap_df_c = pd.DataFrame(data_c, index=merged_c["slice_label"]).T  # constraints × slices

    fig_c = px.imshow(
        heatmap_df_c,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        labels={"color": "gap (violation - global)"},
    )
    fig_c.update_layout(
        height=400,
        xaxis_title=f"Slice ({dim_c})",
        yaxis_title=f"Constraints in layer '{layer_for_constraints}'",
    )
    st.plotly_chart(fig_c, width="stretch")

    # ------------------------------------------------------------------ #
    # BOX PLOT BY DIMENSION
    # ------------------------------------------------------------------ #
    st.subheader("Boxplot by dimension")
    st.markdown(
        """
Select a dimension to see the **distribution of case scores** per category.
This is useful to understand how much variation there is inside a category
and how many outliers it contains.
        """
    )

    if slice_cols:
        dim_box = st.selectbox(
            "Dimension for boxplot",
            slice_cols,
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
        st.plotly_chart(fig_box, width="stretch")
    else:
        st.info("No slice dimensions available for boxplot.")

    # ------------------------------------------------------------------ #
    # SCORES HEATMAP BY DIMENSION (like old cat_dim heatmaps)
    # ------------------------------------------------------------------ #
    st.subheader("Scores heatmap by dimension")
    st.markdown(
        """
Rows = categories of a chosen dimension (e.g. spend areas, vendors).  
Columns = overall WISE score and layer-specific scores.  
Colours = average **score** per category (1 = fully in norm, 0 = worst),
with green = better and red = worse.
        """
    )

    if slice_cols:
        dim_scores = st.selectbox(
            "Dimension for scores heatmap",
            slice_cols,
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
        score_matrix = group_s[metrics].mean()
        score_matrix = score_matrix.rename(columns={"score": "mean_score"})

        fig_scores = px.imshow(
            score_matrix,
            color_continuous_scale="RdYlGn",  # red = low score, green = high score
            aspect="auto",
            labels={"color": "mean score"},
        )
        fig_scores.update_layout(
            height=500,
            xaxis_title="Scores",
            yaxis_title=dim_scores,
        )
        st.plotly_chart(fig_scores, width="stretch")
    else:
        st.info("No slice dimensions available for scores heatmap.")
