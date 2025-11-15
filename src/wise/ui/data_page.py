import streamlit as st
import pandas as pd

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
            st.info("No dataset loaded yet.")
        return

    # Load into DataFrame
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    st.write("Preview (first 10 rows):")
    st.dataframe(df.head(10))

    st.subheader("Map columns")
    cols = list(df.columns)
    case_id_col = st.selectbox("Case ID column", cols)
    activity_col = st.selectbox("Activity column", cols)
    timestamp_col = st.selectbox("Timestamp column", cols)

    slice_candidates = [c for c in cols if c not in [case_id_col, activity_col, timestamp_col]]
    slice_cols = st.multiselect(
        "Slice dimensions (attributes used to group cases)",
        slice_candidates,
    )

    if st.button("Save dataset mapping"):
    # Reset the file pointer before reading again
        uploaded.seek(0)
        df_loaded = load_event_log(
            uploaded,
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
        )
        state.set_dataset_state(
            df=df_loaded,
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
            slice_cols=slice_cols,
        )
        st.success("Dataset mapping saved to session state.")

