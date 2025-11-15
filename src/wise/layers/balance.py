from typing import Optional
import pandas as pd

from .base import BaseLayer
from ..model import Constraint


class BalanceLayer(BaseLayer):
    """L3: Balance – simple relative quantity difference."""
    LAYER_ID = "balance"

    def compute_violation(
        self,
        trace: pd.DataFrame,
        constraint: Constraint,
        activity_col: str,
        timestamp_col: str,
    ) -> float:
        act_from: Optional[str] = constraint.params.get("activity_from")
        act_to: Optional[str] = constraint.params.get("activity_to")
        qty_col_from: Optional[str] = constraint.params.get("qty_col_from")
        qty_col_to: Optional[str] = constraint.params.get("qty_col_to")
        tol: float = float(constraint.params.get("tolerance", 0.02))

        if not (act_from and act_to and qty_col_from and qty_col_to):
            return 0.0

        from_sum = trace.loc[trace[activity_col] == act_from, qty_col_from].sum()
        to_sum = trace.loc[trace[activity_col] == act_to, qty_col_to].sum()
        denom = abs(to_sum) + 1e-6
        rel_diff = abs(from_sum - to_sum) / denom
        return min(1.0, rel_diff / tol)
