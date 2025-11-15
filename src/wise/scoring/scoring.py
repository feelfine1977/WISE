from typing import Dict
import pandas as pd

from ..model import Norm
from ..layers import get_layer
from ..norm import compute_view_weights


def compute_case_scores(
    df: pd.DataFrame,
    norm: Norm,
    view_name: str,
    case_id_col: str,
    activity_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """
    Compute S^{(v)}(sigma) for each case in the log.

    Returns a DataFrame with [case_id_col, "score"].
    """
    weights = compute_view_weights(norm, view_name)
    results = []

    for case_id, trace in df.groupby(case_id_col):
        trace = trace.sort_values(timestamp_col)
        total_violation = 0.0
        for c in norm.constraints:
            layer = get_layer(c.layer_id)
            v = layer.compute_violation(trace, c, activity_col, timestamp_col)
            w = weights[c.id]
            total_violation += w * v
        score = 1.0 - total_violation
        results.append({case_id_col: case_id, "score": score})

    return pd.DataFrame(results)
