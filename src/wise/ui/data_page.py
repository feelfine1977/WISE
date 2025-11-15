import pandas as pd
import streamlit as st

from wise.io.log_loader import load_event_log
from wise.ui import state


def render_data_page():
    st.header("1. Upload event log and define mapping")

    uploaded = st.file_uploader("Upload event log (CSV)", type=["csv"])
    if uploaded is None:
        ds = state.get_dataset_state()
        if ds is not None:
            st.info("Using dataset from session. Upload a new file to replace it.")
            st.dataframe(ds.df.head())
        else:
            st.info("No dataset loaded yet. Upload a CSV file to get started.")
        return

    # Load raw CSV for preview and column selection
    df = pd.read_csv(uploaded)
    st.write("Preview (first 10 rows):")
    st.dataframe(df.head(10))

    cols = list(df.columns)

    def index_or_default(name: str) -> int:
        """Return index of column name if present, else 0."""
        try:
            return cols.index(name)
        except ValueError:
            return 0

    st.subheader("Map columns")

    case_id_col = st.selectbox(
        "Case ID column",
        cols,
        index=index_or_default("case:concept:name"),
        help="Column that identifies individual cases (e.g., PO line id).",
    )
    activity_col = st.selectbox(
        "Activity column",
        cols,
        index=index_or_default("concept:name"),
        help="Column with event / activity labels.",
    )
    timestamp_col = st.selectbox(
        "Timestamp column",
        cols,
        index=index_or_default("time:timestamp"),
        help="Column with event timestamps (will be converted to datetime).",
    )

    # ------------------------------------------------------------------ #
    # Timestamp-derived slice options
    # ------------------------------------------------------------------ #

    st.subheader("Optional derived slice dimensions from timestamps")

    add_dow = st.checkbox(
        "Add slice: day of week (first event per case)",
        value=False,
        help="Creates a column 'slice_first_dow' with Monday, Tuesday, ...",
    )
    add_month = st.checkbox(
        "Add slice: month (first event per case)",
        value=False,
        help="Creates a column 'slice_first_month' (January, February, ...).",
    )
    add_duration_bin = st.checkbox(
        "Add slice: duration category (short/long)",
        value=False,
        help="Computes duration per case and labels each case as short/long.",
    )
    duration_threshold_days = st.number_input(
        "Duration threshold (days) for short/long",
        min_value=0.0,
        value=7.0,
        step=1.0,
        disabled=not add_duration_bin,
        help=(
            "Cases with duration <= threshold are labelled 'short', "
            "cases with duration > threshold are labelled 'long'."
        ),
    )

    # ------------------------------------------------------------------ #
    # Slice dimension selection
    # ------------------------------------------------------------------ #

    st.subheader("Slice dimensions (attributes used to group cases)")

    # Candidate slice columns = all columns except mapped ones
    slice_candidates = [
        c for c in cols if c not in [case_id_col, activity_col, timestamp_col]
    ]

    slice_cols_manual = st.multiselect(
        "Select slice dimensions manually",
        slice_candidates,
        help="These columns will be used to build slices (cohorts) for WISE.",
    )

    st.caption(
        "You can select slice dimensions manually above, or use the automatic "
        "selector below to pick columns that have a limited number of categories."
    )

    max_unique = st.number_input(
        "Max distinct values for auto-selected slice columns",
        min_value=2,
        value=50,
        step=1,
        help=(
            "Columns with at most this many distinct values (excluding NA) "
            "are considered suitable for slicing."
        ),
    )

    # Compute recommended slice columns based on cardinality
    recommended_slices = [
        col
        for col in slice_candidates
        if df[col].nunique(dropna=True) <= max_unique
    ]
    st.write("Auto-suggested slice columns (preview):", recommended_slices)

    use_auto_slices = st.checkbox(
        "Use auto-suggested slice columns instead of manual selection",
        value=False,
        help=(
            "If checked, WISE will use all auto-suggested columns as slice "
            "dimensions (plus any timestamp-derived slices), ignoring the "
            "manual multiselect above."
        ),
    )

    # ------------------------------------------------------------------ #
    # Save mapping
    # ------------------------------------------------------------------ #

    if st.button("Save dataset mapping"):
        # Normalise log (timestamp → datetime, column checks)
        df_loaded = load_event_log(
            df,
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
        )

        # Derived slices based on timestamps
        derived_cols: list[str] = []
        if add_dow or add_month or add_duration_bin:
            grouped = df_loaded.groupby(case_id_col)[timestamp_col]
            first_ts = grouped.min()
            last_ts = grouped.max()

            if add_dow:
                dow = first_ts.dt.day_name()
                col_name = "slice_first_dow"
                df_loaded = df_loaded.merge(
                    dow.rename(col_name),
                    left_on=case_id_col,
                    right_index=True,
                    how="left",
                )
                derived_cols.append(col_name)

            if add_month:
                month = first_ts.dt.month_name()
                col_name = "slice_first_month"
                df_loaded = df_loaded.merge(
                    month.rename(col_name),
                    left_on=case_id_col,
                    right_index=True,
                    how="left",
                )
                derived_cols.append(col_name)

            if add_duration_bin:
                duration_days = (last_ts - first_ts).dt.total_seconds() / 86400.0
                col_name = "slice_duration_cat"
                labels = ["short", "long"]
                bins = [-1e9, duration_threshold_days, 1e9]
                duration_cat = pd.cut(duration_days, bins=bins, labels=labels)
                df_loaded = df_loaded.merge(
                    duration_cat.rename(col_name),
                    left_on=case_id_col,
                    right_index=True,
                    how="left",
                )
                derived_cols.append(col_name)

        # Decide which slice columns to use
        if use_auto_slices:
            chosen_slice_cols = recommended_slices.copy()
        else:
            chosen_slice_cols = list(slice_cols_manual)

        # Always append derived slice columns
        chosen_slice_cols = list(dict.fromkeys(chosen_slice_cols + derived_cols))

        state.set_dataset_state(
            df=df_loaded,
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
            slice_cols=chosen_slice_cols,
        )

        st.success(
            f"Dataset mapping saved. Slice columns: {chosen_slice_cols or 'none'}"
        )
