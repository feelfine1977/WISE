from typing import List, Optional, Dict

import pandas as pd

from wise.layers import get_layer
from wise.model import Norm
from wise.norm import compute_view_weights


def compute_base_violations(
    df: pd.DataFrame,
    norm: Norm,
    case_id_col: str,
    activity_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """
    Compute per-constraint and per-layer violations for each case.

    Returns a DataFrame with columns:
      - case_id_col
      - viol_<constraint_id> for each constraint
      - violation_<layer_id> for each layer (average of its constraint violations)
    """
    layer_ids = sorted({c.layer_id for c in norm.constraints})
    results = []

    for case_id, trace in df.groupby(case_id_col):
        trace = trace.sort_values(timestamp_col)

        row: Dict[str, float] = {case_id_col: case_id}
        layer_sum = {lid: 0.0 for lid in layer_ids}
        layer_cnt = {lid: 0 for lid in layer_ids}

        for c in norm.constraints:
            layer_impl = get_layer(c.layer_id)
            v = layer_impl.compute_violation(trace, c, activity_col, timestamp_col)
            row[f"viol_{c.id}"] = float(v)

            layer_sum[c.layer_id] += v
            layer_cnt[c.layer_id] += 1

        for lid in layer_ids:
            n = layer_cnt[lid]
            row[f"violation_{lid}"] = float(layer_sum[lid] / n) if n > 0 else 0.0

        results.append(row)

    return pd.DataFrame(results)


def add_view_scores(
    base_df: pd.DataFrame,
    norm: Norm,
    views: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Given a base violations DataFrame and a norm, add score_<view> columns.

    base_df must have columns:
      - case_id_col (any name, preserved)
      - viol_<constraint_id> for each constraint in the norm.
    """
    df = base_df.copy()

    # Prepare mapping constraint_id -> viol column
    viol_cols = {c.id: f"viol_{c.id}" for c in norm.constraints}

    if views is None:
        views = [v.name if hasattr(v, "name") else v for v in norm.views]

    for view_name in views:
        weights = compute_view_weights(norm, view_name)
        total = 0.0
        for cid, col in viol_cols.items():
            if col not in df.columns:
                continue
            total = total + weights[cid] * df[col]
        df[f"score_{view_name}"] = 1.0 - total

    return df


def precompute_scores_for_all_views(
    df: pd.DataFrame,
    norm: Norm,
    case_id_col: str,
    activity_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """
    Convenience function: base violations + score_<view> for all norm.views.
    """
    base = compute_base_violations(
        df=df,
        norm=norm,
        case_id_col=case_id_col,
        activity_col=activity_col,
        timestamp_col=timestamp_col,
    )
    return add_view_scores(base, norm)
