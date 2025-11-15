from typing import Optional
import pandas as pd

from .base import BaseLayer
from ..model import Constraint


class PresenceLayer(BaseLayer):
    """
    L1: Presence – check that a given activity appears at least once.
    """
    LAYER_ID = "presence"

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
        return 0.0 if activity in trace[activity_col].values else 1.0
