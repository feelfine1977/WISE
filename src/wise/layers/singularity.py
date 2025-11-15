from typing import Optional
import pandas as pd

from .base import BaseLayer
from ..model import Constraint


class SingularityLayer(BaseLayer):
    """L4: Singularity – penalise repeated occurrences."""
    LAYER_ID = "singularity"

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
        k = (trace[activity_col] == activity).sum()
        if k <= 1:
            return 0.0
        # simple: at least one repeat → full violation
        return 1.0
