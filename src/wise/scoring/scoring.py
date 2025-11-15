from typing import Dict

import pandas as pd

from wise.model import Norm
from wise.layers import get_layer
from wise.norm import compute_view_weights


def compute_case_scores(
    df: pd.DataFrame,
    norm: Norm,
    view_name: str,
    case_id_col: str,
    activity_col: str,
    timestamp_col: str,
    weights_override: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute WISE scores for each case and record per-layer & per-constraint violations.

    Returns a DataFrame with columns:
      - [case_id_col]
      - "score"                        : overall WISE score in [0, 1]
      - "violation_<layer_id>"        : mean violation for each layer
      - "viol_<constraint_id>"        : violation per constraint (0..1)

    Parameters
    ----------
    df : DataFrame
        Event log.
    norm : Norm
        Process norm with constraints and views.
    view_name : str
        Name of the view to use (e.g. "Finance").
    case_id_col, activity_col, timestamp_col : str
        Column names.
    weights_override : dict, optional
        Optional dict {constraint_id: weight} (already normalised).
        If None, weights are computed from the view + norm.

    Notes
    -----
    - Violations v_c(sigma) are in [0,1].
    - Score S^{(v)}(sigma) = 1 - sum_c w_c^{(v)} * v_c(sigma).
    """
    # Determine weights per constraint
    if weights_override is None:
        weights = compute_view_weights(norm, view_name)
    else:
        weights = weights_override

    layer_ids = sorted({c.layer_id for c in norm.constraints})

    results = []

    for case_id, trace in df.groupby(case_id_col):
        trace = trace.sort_values(timestamp_col)

        # Per-layer accumulators
        layer_violation_sum: Dict[str, float] = {lid: 0.0 for lid in layer_ids}
        layer_violation_cnt: Dict[str, int] = {lid: 0 for lid in layer_ids}

        # Per-constraint violations
        constraint_viol: Dict[str, float] = {}

        total_violation_weighted = 0.0

        for c in norm.constraints:
            layer_impl = get_layer(c.layer_id)
            v = layer_impl.compute_violation(trace, c, activity_col, timestamp_col)
            w = weights[c.id]

            total_violation_weighted += w * v

            layer_violation_sum[c.layer_id] += v
            layer_violation_cnt[c.layer_id] += 1

            constraint_viol[f"viol_{c.id}"] = v

        # Average violation per layer (unweighted, for inspection)
        layer_violation_avg: Dict[str, float] = {}
        for lid in layer_ids:
            n = layer_violation_cnt[lid]
            layer_violation_avg[f"violation_{lid}"] = (
                layer_violation_sum[lid] / n if n > 0 else 0.0
            )

        score = 1.0 - total_violation_weighted

        row = {
            case_id_col: case_id,
            "score": score,
        }
        row.update(layer_violation_avg)
        row.update(constraint_viol)
        results.append(row)

    return pd.DataFrame(results)
