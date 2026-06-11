"""
wise.norm — Phase 1: instantiate norm, layers, and views (Fig. 2, green band).

A process norm is a triple  N = (C, Λ, lay)  (paper Sec. IV-C):
  * C    — finite set of machine-checkable constraints,
  * Λ    — finite set of business-facing layers,
  * lay  — assigns each constraint to exactly one layer
           (so {C_λ} partitions C).

Layers are not extra constraints — they group constraints so a slice can
be *explained* by stating which deviation mechanisms dominate, instead of
listing dozens of isolated violations.

A view p is a non-negative raw weight vector ⟨w_c^(p)⟩ over the *shared*
constraint set. Views do not define different norms — they define
different priorities over the same norm, which is what lets the same
case evidence answer Finance, Logistics, Compliance, and Automation
questions without redefining any check.

Two-stage weight elicitation (paper Sec. IV-C, used for BPIC'19):
    1. layer weights      a_λ^(p) ≥ 0,  Σ_λ a_λ^(p) = 1   (per view)
    2. within-layer       b̂_c = b_c / Σ_{d∈C_λ} b_d        (shared)
    →  raw weight         w_c^(p) = a_{lay(c)}^(p) · b̂_c
"""
import json
from pathlib import Path

import pandas as pd

from .config import CONFIG


def load_norm(path=None) -> dict:
    """Load the versioned norm artefact (governed JSON file)."""
    path = Path(path or CONFIG["norm_path"])
    with open(path) as f:
        norm = json.load(f)
    return norm


def layer_weight_table(norm: dict) -> pd.DataFrame:
    """Paper Table VIII — layer-level role weights a_λ^(p), one column per view."""
    rows = {}
    for view, info in norm["views"].items():
        rows[view] = pd.Series(info["layer_weights"])
    tab = pd.DataFrame(rows)
    tab.index.name = "layer_id"
    tab["layer_name"] = [norm["layers"][l]["name"] for l in tab.index]
    return tab[["layer_name"] + list(norm["views"].keys())]


def within_layer_weights(norm: dict) -> pd.DataFrame:
    """Normalised within-layer weights b̂_c (shared across views)."""
    rows = []
    for layer_id in norm["layers"]:
        lcs = [c for c in norm["constraints"] if c["layer_id"] == layer_id]
        total = sum(c.get("within_layer_weight", 1.0) for c in lcs)
        for c in lcs:
            b = c.get("within_layer_weight", 1.0)
            rows.append({
                "layer_id": layer_id,
                "constraint_id": c["id"],
                "paper_type": c.get("paper_type", c["type"]),
                "b_c (raw)": b,
                "b̂_c (normalised)": b / total,
                "description": c["description"],
            })
    return pd.DataFrame(rows)


def raw_constraint_weights(norm: dict) -> pd.DataFrame:
    """Final raw constraint weights  w_c^(p) = a_λ^(p) · b̂_c  per view.

    These are the only quantities the scoring equations consume — the
    two-stage construction above is a convenience for elicitation and
    governance, not part of the maths (paper: "the scoring equations use
    only the resulting raw weights and do not depend on how they were
    obtained").
    """
    wl = within_layer_weights(norm).set_index("constraint_id")
    out = wl[["layer_id", "paper_type", "b̂_c (normalised)"]].copy()
    for view, info in norm["views"].items():
        a = info["layer_weights"]
        out[f"w^({view})"] = [
            a[wl.loc[cid, "layer_id"]] * wl.loc[cid, "b̂_c (normalised)"]
            for cid in out.index
        ]
    return out
