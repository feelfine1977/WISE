import streamlit as st

from wise.ui import state
from wise.ui.data_page import render_data_page
from wise.ui.norm_page import render_norm_page
from wise.ui.results_page import render_results_page



def main():
    st.set_page_config(page_title="WISE", layout="wide")
    st.title("WISE – Norm-based Prioritization")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "1. Data & Mapping",
            "2. Norm",
            "3. Results",
        ],
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Reset WISE state"):
        state.reset_all()
        st.experimental_rerun()

    if page.startswith("1."):
        render_data_page()
    elif page.startswith("2."):
        render_norm_page()
    elif page.startswith("3."):
        render_results_page()


if __name__ == "__main__":
    main()
