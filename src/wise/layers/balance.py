# src/wise/layers/balance.py

from typing import Any, Dict

import pandas as pd

from wise.layers.base import BaseLayer


class BalanceLayer(BaseLayer):
    """
    L3: Balance layer.

    Checks whether numeric quantities / amounts from two activities
    are approximately balanced within a given tolerance.

    Constraint params:
      - activity_from: str
      - activity_to: str
      - qty_col_from: str
      - qty_col_to: str
      - tolerance: float in [0,1]  (relative tolerance, e.g. 0.05 for 5%)
    """

    LAYER_ID = "balance"

    def compute_violation(
        self,
        trace: pd.DataFrame,
        constraint: Any,
        activity_col: str,
        timestamp_col: str,
    ) -> float:
        # Get params from Constraint object or dict
        params: Dict = getattr(constraint, "params", None) or getattr(constraint, "params_", None) or {}
        if not params and isinstance(constraint, dict):
            params = constraint.get("params", {}) or {}

        act_from = params.get("activity_from")
        act_to = params.get("activity_to")
        qty_col_from = params.get("qty_col_from")
        qty_col_to = params.get("qty_col_to")
        tol = float(params.get("tolerance", 0.0))

        # If any critical parameter is missing, do not penalize (other layers handle this)
        if not act_from or not act_to or not qty_col_from or not qty_col_to:
            return 0.0

        cols = trace.columns

        # If the configured quantity columns are not present in the log, skip this constraint.
        if qty_col_from not in cols or qty_col_to not in cols:
            # This typically means the norm was defined for a different schema.
            # Returning 0 keeps WISE running; presence/order layers still capture issues.
            return 0.0

        # Select events for each side
        from_mask = trace[activity_col] == act_from
        to_mask = trace[activity_col] == act_to

        # If either side is completely missing, leave it to presence/order layers
        if not from_mask.any() or not to_mask.any():
            return 0.0

        from_sum = trace.loc[from_mask, qty_col_from].fillna(0).sum()
        to_sum = trace.loc[to_mask, qty_col_to].fillna(0).sum()

        # If both sums are zero, there is nothing to balance
        if from_sum == 0 and to_sum == 0:
            return 0.0

        # Relative difference
        denom = max(abs(to_sum), abs(from_sum), 1e-6)
        rel_diff = abs(from_sum - to_sum) / denom

        # No tolerance: pure relative difference, capped at 1
        if tol <= 0:
            return float(min(1.0, rel_diff))

        # With tolerance: ignore differences within [0, tol], rescale above tol into [0, 1]
        if rel_diff <= tol:
            return 0.0
        return float(min(1.0, (rel_diff - tol) / (1.0 - tol)))
