from typing import Optional
import pandas as pd

from .base import BaseLayer
from ..model import Constraint


class ExclusionLayer(BaseLayer):
    """L5: Exclusion – forbidden activity should not occur."""
    LAYER_ID = "exclusion"

    def compute_violation(
        self,
        trace: pd.DataFrame,
        constraint: Constraint,
        activity_col: str,
        timestamp_col: str,
    ) -> float:
        activity: Optional[str] = constraint.params.get("activity")
        if not activity:
            return 0.0
        return 1.0 if activity in trace[activity_col].values else 0.0
