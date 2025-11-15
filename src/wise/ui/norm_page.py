import json
from typing import Dict, List

import pandas as pd
import streamlit as st

from wise.io.norm_loader import load_norm_from_json
from wise.model import Constraint, Norm, View
from wise.ui import state


# Keys for Streamlit session_state
BUILDER_VIEWS_KEY = "norm_builder_views"
BUILDER_CONSTRAINTS_KEY = "norm_builder_constraints"
BUILDER_WEIGHTS_KEY = "norm_builder_weights_df"
BUILDER_NAME_KEY = "norm_builder_name"
BUILDER_NEXT_ID_KEY = "norm_builder_next_id"


# ------------------------------------------------------------------ #
# Builder state helpers
# ------------------------------------------------------------------ #

def _init_builder_state():
    """Initialise norm builder state in session if not present."""
    if BUILDER_VIEWS_KEY not in st.session_state:
        st.session_state[BUILDER_VIEWS_KEY] = ["Finance", "Logistics"]
    if BUILDER_CONSTRAINTS_KEY not in st.session_state:
        st.session_state[BUILDER_CONSTRAINTS_KEY] = []
    if BUILDER_WEIGHTS_KEY not in st.session_state:
        st.session_state[BUILDER_WEIGHTS_KEY] = None
    if BUILDER_NAME_KEY not in st.session_state:
        st.session_state[BUILDER_NAME_KEY] = "Interactive WISE norm"
    if BUILDER_NEXT_ID_KEY not in st.session_state:
        st.session_state[BUILDER_NEXT_ID_KEY] = 1


def _new_constraint_id() -> str:
    idx = st.session_state[BUILDER_NEXT_ID_KEY]
    st.session_state[BUILDER_NEXT_ID_KEY] = idx + 1
    return f"c{idx:03d}"


def _get_builder_views() -> List[str]:
    return st.session_state[BUILDER_VIEWS_KEY]


def _get_builder_constraints() -> List[Dict]:
    return st.session_state[BUILDER_CONSTRAINTS_KEY]


def _set_builder_constraints(constraints: List[Dict]):
    st.session_state[BUILDER_CONSTRAINTS_KEY] = constraints
    # Constraints changed → weight table must be rebuilt
    st.session_state[BUILDER_WEIGHTS_KEY] = None


def _populate_from_norm(norm: Norm):
    """Load an existing Norm into the builder state."""
    st.session_state[BUILDER_VIEWS_KEY] = [v.name for v in norm.views]
    st.session_state[BUILDER_CONSTRAINTS_KEY] = []
    for c in norm.constraints:
        st.session_state[BUILDER_CONSTRAINTS_KEY].append(
            {
                "id": c.id,
                "layer_id": c.layer_id,
                "params": c.params,
                "base_weight": c.base_weight,
                "view_weights": c.view_weights,
            }
        )
    st.session_state[BUILDER_NAME_KEY] = norm.metadata.get("name", "Loaded norm")
    st.session_state[BUILDER_WEIGHTS_KEY] = None
    st.session_state[BUILDER_NEXT_ID_KEY] = len(norm.constraints) + 1


def _describe_constraint(c: Dict) -> str:
    """Return a human-readable description for a constraint dict."""
    layer = c["layer_id"]
    p = c.get("params", {})

    if layer == "presence":
        act = p.get("activity", "?")
        return f"{act!r} must occur at least once per case"
    if layer == "order_lag":
        a1 = p.get("activity_from", "?")
        a2 = p.get("activity_to", "?")
        d = p.get("max_days")
        if d is not None:
            return f"{a1!r} should be followed by {a2!r} within {d} days"
        return f"{a1!r} should be followed by {a2!r}"
    if layer == "balance":
        a1 = p.get("activity_from", "?")
        a2 = p.get("activity_to", "?")
        q1 = p.get("qty_col_from", "?")
        q2 = p.get("qty_col_to", "?")
        tol = p.get("tolerance", 0.0)
        return (
            f"Quantities of {a1!r}.{q1!r} should match {a2!r}.{q2!r} "
            f"within ±{int(tol * 100)}%"
        )
    if layer == "singularity":
        act = p.get("activity", "?")
        return f"{act!r} should not repeat (at most once per case)"
    if layer == "exclusion":
        act = p.get("activity", "?")
        return f"{act!r} should not occur"

    return f"{layer}: {p}"


# ------------------------------------------------------------------ #
# Main page
# ------------------------------------------------------------------ #

def render_norm_page():
    st.header("2. Build or load a process norm")

    _init_builder_state()

    ds = state.get_dataset_state()
    if ds is None:
        st.info("Upload and map data first on the 'Data & Mapping' page.")
        return

    activities = sorted(ds.df[ds.activity_col].dropna().unique().tolist())
    numeric_cols = sorted(
        [c for c in ds.df.columns if pd.api.types.is_numeric_dtype(ds.df[c])]
    )

    tab_views, tab_constraints, tab_weights = st.tabs(
        ["1. Views", "2. Constraints", "3. Weights & export"]
    )

    # ------------------------ TAB 1: Views ---------------------------- #
    with tab_views:
        st.subheader("Views")

        st.markdown(
            """
Views represent stakeholder perspectives, e.g. **Finance**, **Logistics**,
**Automation**. Each view can assign different weights to the same constraints.
            """
        )

        views = _get_builder_views()

        new_view = st.text_input(
            "Add new view",
            placeholder="e.g. Compliance",
            key="norm_add_view",
        )
        col_add, col_clear = st.columns([1, 1])
        with col_add:
            if st.button("Add view"):
                if new_view and new_view not in views:
                    views.append(new_view)
                    st.session_state[BUILDER_VIEWS_KEY] = views
                    st.session_state[BUILDER_WEIGHTS_KEY] = None
        with col_clear:
            if st.button("Reset views to default"):
                st.session_state[BUILDER_VIEWS_KEY] = ["Finance", "Logistics"]
                st.session_state[BUILDER_WEIGHTS_KEY] = None

        if views:
            st.write("Current views:", views)
        else:
            st.warning("No views defined yet. Add at least one view before weighting.")

        st.markdown("---")
        st.subheader("Load existing norm (optional)")

        uploaded = st.file_uploader("Load norm JSON", type=["json"])
        if uploaded is not None:
            try:
                norm = load_norm_from_json(uploaded)
            except Exception as e:
                st.error(f"Could not load norm: {e}")
            else:
                _populate_from_norm(norm)
                state.set_norm_state(norm)
                st.success("Norm loaded into builder and set as current WISE norm.")
                st.json(
                    {
                        "views": [v.name for v in norm.views],
                        "n_constraints": len(norm.constraints),
                    }
                )

    # -------------------- TAB 2: Constraints -------------------------- #
    with tab_constraints:
        st.subheader("Constraints")

        st.markdown(
            """
Build constraints per layer. The activity names below are taken from the
uploaded event log.
            """
        )

        constraints = _get_builder_constraints()

        # Presence layer
        with st.expander("Presence layer (L1) – required activities"):
            pres_acts = st.multiselect(
                "Activities that must appear at least once per case",
                activities,
                key="pres_acts",
            )
            if st.button("Add presence constraints", key="btn_add_presence"):
                for act in pres_acts:
                    constraints.append(
                        {
                            "id": _new_constraint_id(),
                            "layer_id": "presence",
                            "params": {"activity": act},
                            "base_weight": 1.0,
                            "view_weights": {},
                        }
                    )
                _set_builder_constraints(constraints)
                st.success(f"Added {len(pres_acts)} presence constraints.")

        # Order/lag layer
        with st.expander("Order/lag layer (L2) – sequence and time between activities"):
            col_from, col_to, col_days = st.columns([2, 2, 1])
            with col_from:
                ord_from = st.selectbox(
                    "From activity",
                    activities,
                    key="ord_from",
                )
            with col_to:
                ord_to = st.selectbox(
                    "To activity",
                    activities,
                    key="ord_to",
                )
            with col_days:
                max_days = st.number_input(
                    "Max days",
                    min_value=0.0,
                    value=10.0,
                    step=1.0,
                    key="ord_max_days",
                )
            if st.button("Add order/lag constraint", key="btn_add_order"):
                constraints.append(
                    {
                        "id": _new_constraint_id(),
                        "layer_id": "order_lag",
                        "params": {
                            "activity_from": ord_from,
                            "activity_to": ord_to,
                            "max_days": max_days,
                        },
                        "base_weight": 1.0,
                        "view_weights": {},
                    }
                )
                _set_builder_constraints(constraints)
                st.success("Added order/lag constraint.")

        # Balance layer
        with st.expander("Balance layer (L3) – quantities should match"):
            st.caption(
                "Use this to express expectations like: total GR quantity should "
                "match total Invoice quantity (within a tolerance)."
            )
            if not numeric_cols:
                st.info("No numeric columns found in the log; balance constraints may be limited.")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                bal_from = st.selectbox(
                    "From activity",
                    activities,
                    key="bal_from",
                )
                qty_from = st.selectbox(
                    "Quantity column (from)",
                    numeric_cols or ["<no numeric columns>"],
                    key="bal_qty_from",
                )
            with col_b2:
                bal_to = st.selectbox(
                    "To activity",
                    activities,
                    key="bal_to",
                )
                qty_to = st.selectbox(
                    "Quantity column (to)",
                    numeric_cols or ["<no numeric columns>"],
                    key="bal_qty_to",
                )
            tol_percent = st.number_input(
                "Relative tolerance (%)",
                min_value=0.0,
                value=5.0,
                step=1.0,
                key="bal_tol",
                help="Allowed relative difference between sums (e.g., 5% = ±0.05).",
            )
            if st.button("Add balance constraint", key="btn_add_balance"):
                if numeric_cols:
                    constraints.append(
                        {
                            "id": _new_constraint_id(),
                            "layer_id": "balance",
                            "params": {
                                "activity_from": bal_from,
                                "activity_to": bal_to,
                                "qty_col_from": qty_from,
                                "qty_col_to": qty_to,
                                "tolerance": tol_percent / 100.0,
                            },
                            "base_weight": 1.0,
                            "view_weights": {},
                        }
                    )
                    _set_builder_constraints(constraints)
                    st.success("Added balance constraint.")
                else:
                    st.warning("Cannot add balance constraint: no numeric columns available.")

        # Singularity layer
        with st.expander("Singularity layer (L4) – avoid rework (repeated events)"):
            sing_acts = st.multiselect(
                "Activities that should occur at most once per case",
                activities,
                key="sing_acts",
            )
            if st.button("Add singularity constraints", key="btn_add_singularity"):
                for act in sing_acts:
                    constraints.append(
                        {
                            "id": _new_constraint_id(),
                            "layer_id": "singularity",
                            "params": {"activity": act},
                            "base_weight": 1.0,
                            "view_weights": {},
                        }
                    )
                _set_builder_constraints(constraints)
                st.success(f"Added {len(sing_acts)} singularity constraints.")

        # Exclusion layer
        with st.expander("Exclusion layer (L5) – forbidden activities"):
            excl_acts = st.multiselect(
                "Activities that should not occur",
                activities,
                key="excl_acts",
            )
            if st.button("Add exclusion constraints", key="btn_add_exclusion"):
                for act in excl_acts:
                    constraints.append(
                        {
                            "id": _new_constraint_id(),
                            "layer_id": "exclusion",
                            "params": {"activity": act},
                            "base_weight": 1.0,
                            "view_weights": {},
                        }
                    )
                _set_builder_constraints(constraints)
                st.success(f"Added {len(excl_acts)} exclusion constraints.")

        st.markdown("---")
        st.subheader("Current constraints")

        if constraints:
            df_c = pd.DataFrame(
                [
                    {
                        "id": c["id"],
                        "layer_id": c["layer_id"],
                        "description": _describe_constraint(c),
                        "params": c["params"],
                    }
                    for c in constraints
                ]
            )
            st.dataframe(df_c)
            if st.button("Clear all constraints"):
                _set_builder_constraints([])
                st.success("All constraints removed.")
        else:
            st.info("No constraints defined yet.")

    # ---------------- TAB 3: Weights & export ------------------------- #
    with tab_weights:
        st.subheader("Weights & export")

        norm_name = st.text_input(
            "Norm name",
            value=st.session_state[BUILDER_NAME_KEY],
            key="norm_name_input",
        )
        st.session_state[BUILDER_NAME_KEY] = norm_name

        views = _get_builder_views()
        constraints = _get_builder_constraints()

        if not views:
            st.warning("Define at least one view on the 'Views' tab.")
            return
        if not constraints:
            st.warning("Define at least one constraint on the 'Constraints' tab.")
            return

        # Initialise or refresh weight table if needed
        weights_df = st.session_state[BUILDER_WEIGHTS_KEY]
        ids = [c["id"] for c in constraints]

        if (
            weights_df is None
            or list(weights_df.index) != ids
            or list(weights_df.columns) != views
        ):
            weights_df = pd.DataFrame(1.0, index=ids, columns=views)
            st.session_state[BUILDER_WEIGHTS_KEY] = weights_df

        st.markdown(
            """
Edit the weights per constraint and view below.

Each cell is a value between **0.0** and **1.0** in steps of **0.1**:

- `0.0` → this constraint does not contribute in that view;
- `1.0` → full contribution;
- values in-between → intermediate importance.
            """
        )

        weight_options = [round(x * 0.1, 1) for x in range(0, 11)]

        column_config = {
            view: st.column_config.SelectboxColumn(
                options=weight_options,
                default=1.0,
                width="small",
                help=f"Weight of constraint in view '{view}' (0.0–1.0 in steps of 0.1).",
            )
            for view in views
        }

        edited_df = st.data_editor(
            st.session_state[BUILDER_WEIGHTS_KEY],
            column_config=column_config,
            num_rows="fixed",
            width="stretch",
            key="weights_editor",
        )
        st.session_state[BUILDER_WEIGHTS_KEY] = edited_df

        col_use, col_download = st.columns([1, 1])

        with col_use:
            if st.button("Use this norm in WISE"):
                norm = _build_norm_from_builder()
                state.set_norm_state(norm)
                st.success(
                    f"Norm '{norm.metadata.get('name', '')}' set as current WISE norm "
                    f"with {len(norm.constraints)} constraints and views {views}."
                )

        with col_download:
            norm = _build_norm_from_builder()
            export_dict = {
                "metadata": norm.metadata,
                "views": [v.name for v in norm.views],
                "constraints": [
                    {
                        "id": c.id,
                        "layer_id": c.layer_id,
                        "params": c.params,
                        "base_weight": c.base_weight,
                        "view_weights": c.view_weights,
                    }
                    for c in norm.constraints
                ],
            }
            json_str = json.dumps(export_dict, indent=2)
            st.download_button(
                "Download norm as JSON",
                data=json_str,
                file_name="WISE_norm.json",
                mime="application/json",
            )

        st.markdown("---")
        st.subheader("Preview of exported norm")
        st.json(
            {
                "name": st.session_state[BUILDER_NAME_KEY],
                "views": views,
                "n_constraints": len(constraints),
            }
        )


def _build_norm_from_builder() -> Norm:
    """Construct a Norm object from the builder state."""
    views = [View(name=v) for v in st.session_state[BUILDER_VIEWS_KEY]]
    constraints_raw: List[Dict] = st.session_state[BUILDER_CONSTRAINTS_KEY]
    weights_df: pd.DataFrame = st.session_state[BUILDER_WEIGHTS_KEY]
    norm_name = st.session_state[BUILDER_NAME_KEY]

    constraints: List[Constraint] = []
    for cr in constraints_raw:
        cid = cr["id"]
        view_weights = (
            weights_df.loc[cid].to_dict() if cid in weights_df.index else {}
        )
        base_weight = float(
            weights_df.loc[cid].mean()
        ) if cid in weights_df.index else 1.0

        constraints.append(
            Constraint(
                id=cid,
                layer_id=cr["layer_id"],
                params=cr["params"],
                base_weight=base_weight,
                view_weights=view_weights,
            )
        )

    return Norm(
        constraints=constraints,
        views=views,
        metadata={"name": norm_name},
    )
