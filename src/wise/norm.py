from typing import Dict
from .model import Norm, Constraint, View


def compute_view_weights(norm: Norm, view_name: str) -> Dict[str, float]:
    """
    Compute normalised weights w_c^{(v)} for a given view.

    If a constraint has no view-specific weight for this view, use base_weight.
    If all weights are zero, fall back to equal weights.
    """
    raw: Dict[str, float] = {}
    for c in norm.constraints:
        view_weight = c.view_weights.get(view_name, c.base_weight)
        raw[c.id] = max(0.0, float(view_weight))

    total = sum(raw.values())
    if total <= 0:
        n = len(raw) or 1
        return {cid: 1.0 / n for cid in raw}

    return {cid: w / total for cid, w in raw.items()}


def add_view(norm: Norm, name: str) -> Norm:
    """Utility to append a view if not present."""
    if name not in norm.get_view_names():
        norm.views.append(View(name=name))
    return norm
