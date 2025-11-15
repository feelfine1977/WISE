from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
from ..model import Constraint


class BaseLayer(ABC):
    """
    Abstract base class for a layer.

    Each concrete layer provides:
    - LAYER_ID: string identifier used in norm.layer_id
    - compute_violation(): v_c(sigma) in [0,1] for one case & one constraint.
    """

    LAYER_ID: str  # e.g. "presence"

    @abstractmethod
    def compute_violation(
        self,
        trace: pd.DataFrame,
        constraint: Constraint,
        activity_col: str,
        timestamp_col: str,
    ) -> float:
        """Compute violation for a single case and constraint."""
        ...
