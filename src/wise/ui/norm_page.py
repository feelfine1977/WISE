import json
from io import StringIO
from typing import Optional

import streamlit as st

from wise.io.norm_loader import load_norm_from_json
from wise.model import Norm, View
from wise.ui import state


def render_norm_page():
    st.header("2. Load or define process norm")

    norm = state.get_norm_state()
    if norm is not None:
        st.success("A norm is already loaded in session.")
        _show_norm_summary(norm)

    st.subheader("Load norm from JSON")
    uploaded = st.file_uploader("Norm JSON file", type=["json"])
    if uploaded is not None:
        try:
            norm = load_norm_from_json(uploaded)
        except Exception as e:
            st.error(f"Could not load norm: {e}")
            return

        # Ensure views list is not empty; if raw file only had string views
        if not norm.views and hasattr(norm, "metadata"):
            # already handled by loader, but keep here as sanity
            pass

        state.set_norm_state(norm)
        st.success("Norm loaded and saved to session.")
        _show_norm_summary(norm)

    st.markdown("---")
    st.subheader("Create a very simple norm inline (demo)")
    st.caption("This is just a tiny helper; for real projects define norms in JSON.")

    if st.button("Create demo norm for current dataset"):
        ds = state.get_dataset_state()
        if ds is None:
            st.warning("Upload data first on the 'Data & Mapping' page.")
            return
        demo_norm = _create_demo_presence_norm(ds)
        state.set_norm_state(demo_norm)
        st.success("Demo presence-based norm created.")
        _show_norm_summary(demo_norm)


def _show_norm_summary(norm: Norm):
    st.write("Views:", [v.name for v in norm.views])
    st.write("Number of constraints:", len(norm.constraints))
    if len(norm.constraints) > 0:
        st.write("First few constraints:")
        st.json([
            {
                "id": c.id,
                "layer_id": c.layer_id,
                "params": c.params,
            }
            for c in norm.constraints[:5]
        ])


def _create_demo_presence_norm(ds: state.DatasetState) -> Norm:
    """
    Very simple norm: Presence constraints for the top activities.
    """
    from wise.model import Constraint, Norm, View

    top_activities = (
        ds.df[ds.activity_col]
        .value_counts()
        .head(5)
        .index.tolist()
    )

    constraints = []
    for i, act in enumerate(top_activities, start=1):
        constraints.append(
            Constraint(
                id=f"c_presence_{i}",
                layer_id="presence",
                params={"activity": act},
                base_weight=1.0,
                view_weights={"Default": 1.0},
            )
        )

    return Norm(
        constraints=constraints,
        views=[View(name="Default")],
        metadata={"name": "Demo presence norm"},
    )
