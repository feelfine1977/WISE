import streamlit as st

from wise.scoring.scoring import compute_case_scores
from wise.scoring.slices import aggregate_slices
from wise.ui import state


def render_results_page():
    st.header("3. Run WISE and inspect results")

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
    shrink_k = st.slider("Shrinkage k", 0.0, 200.0, 50.0, step=10.0)

    if st.button("Compute scores and priorities"):
        with st.spinner("Computing case scores..."):
            case_scores = compute_case_scores(
                df=ds.df,
                norm=norm,
                view_name=view_name,
                case_id_col=ds.case_id_col,
                activity_col=ds.activity_col,
                timestamp_col=ds.timestamp_col,
            )

        st.subheader("Case scores (sample)")
        st.dataframe(case_scores.head(20))

        if not ds.slice_cols:
            st.info("No slice dimensions defined; only case scores are available.")
            state.set_results_state(view_name, case_scores, slice_summary=None, params={"shrink_k": shrink_k})
            return

        with st.spinner("Aggregating slices..."):
            slice_summary = aggregate_slices(
                df_scores=case_scores,
                df_log=ds.df,
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

        st.subheader("Slice-level priorities")
        st.dataframe(slice_summary.head(50))

        # Very simple bar chart by PI – you can switch to heatmaps later
        st.subheader("Top slices by PI")
        st.bar_chart(slice_summary.set_index(ds.slice_cols)["PI"].head(20))
    else:
        # If results already exist in session, show a summary
        results = state.get_results_state()
        if results is not None:
            st.info(f"Showing results for view '{results.view_name}' (from session).")
            st.dataframe(results.slice_summary.head(50))
        else:
            st.info("Press 'Compute scores and priorities' to run WISE.")
