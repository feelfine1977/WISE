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
    Compute S^{(v)}(sigma) for each case in the log.

    Returns a DataFrame with:
      - case_id_col
      - "score"
      - "violation_<layer_id>" columns (mean violation per layer)
      - "viol_<constraint_id>" columns (violation per constraint)

    weights_override (optional) can be used to pass custom constraint weights
    (already normalised); if None, view-based weights are computed.
    """
    # determine weights per constraint
    if weights_override is None:
        weights = compute_view_weights(norm, view_name)
    else:
        weights = weights_override

    layer_ids = sorted({c.layer_id for c in norm.constraints})

    results = []
    for case_id, trace in df.groupby(case_id_col):
        trace = trace.sort_values(timestamp_col)

        # layer accumulators
        layer_violation_sum: Dict[str, float] = {lid: 0.0 for lid in layer_ids}
        layer_violation_cnt: Dict[str, int] = {lid: 0 for lid in layer_ids}

        # per-constraint violations
        constraint_viol: Dict[str, float] = {}

        total_violation_weighted = 0.0

        for c in norm.constraints:
            layer_impl = get_layer(c.layer_id)
            v = layer_impl.compute_violation(trace, c, activity_col, timestamp_col)
            w = weights[c.id]

            total_violation_weighted += w * v

            layer_violation_sum[c.layer_id] += v
            layer_violation_cnt[c.layer_id] += 1

            # store per-constraint violation (0..1)
            constraint_viol[f"viol_{c.id}"] = v

        # avg violation per layer (unweighted, for inspection)
        layer_violation_avg = {}
        for lid in layer_ids:
            n = layer_violation_cnt[lid]
            layer_violation_avg[f"violation_{lid}"] = (
                layer_violation_sum[lid] / n if n > 0 else 0.0
            )

        score = 1.0 - total_violation_weighted

        results.append(
            {
                case_id_col: case_id,
                "score": score,
                **layer_violation_avg,
                **constraint_viol,
            }
        )

    return pd.DataFrame(results)
