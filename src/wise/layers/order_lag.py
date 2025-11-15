from typing import Optional
import pandas as pd

from .base import BaseLayer
from ..model import Constraint


class OrderLagLayer(BaseLayer):
    """
    L2: Order / lag – checks basic ordering and optional lag threshold.
    """
    LAYER_ID = "order_lag"

    def compute_violation(
        self,
        trace: pd.DataFrame,
        constraint: Constraint,
        activity_col: str,
        timestamp_col: str,
    ) -> float:
        act_from: Optional[str] = constraint.params.get("activity_from")
        act_to: Optional[str] = constraint.params.get("activity_to")
        max_days: Optional[float] = constraint.params.get("max_days")

        if not act_from or not act_to:
            return 0.0

        # sort by time for safety
        trace = trace.sort_values(timestamp_col)
        if act_from not in trace[activity_col].values or act_to not in trace[activity_col].values:
            return 1.0

        from_idx = trace.index[trace[activity_col] == act_from][0]
        to_idx = trace.index[trace[activity_col] == act_to][0]
        t_from = trace.loc[from_idx, timestamp_col]
        t_to = trace.loc[to_idx, timestamp_col]

        if t_from > t_to:
            return 1.0  # wrong order

        if max_days is None:
            return 0.0

        lag_days = (t_to - t_from).days
        if lag_days <= max_days:
            return 0.0

        return min(1.0, (lag_days - max_days) / max_days)
