import streamlit as st

from wise.ui import state
from wise.ui.data_page import render_data_page
from wise.ui.norm_page import render_norm_page
from wise.ui.results_page import render_results_page
from wise.ui.refinement_page import render_refinement_page


def main():
    st.set_page_config(
        page_title="WISE – Norm-based Prioritization",
        layout="wide",
    )

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Data & Mapping", "Norm", "Results", "Refinement"],
    )

    if st.sidebar.button("Reset WISE state"):
        state.reset_all()
        st.rerun()

    if page == "Data & Mapping":
        render_data_page()
    elif page == "Norm":
        render_norm_page()
    elif page == "Results":
        render_results_page()
    else:
        render_refinement_page()


if __name__ == "__main__":
    main()
