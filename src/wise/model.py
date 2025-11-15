from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Constraint:
    """
    One constraint in the process norm.

    layer_id: identifier of the layer implementation, e.g. "presence", "order_lag".
    params: layer-specific configuration, e.g. {"activity": "Record Goods Receipt"}.
    base_weight: default weight if no view-specific weight is given.
    view_weights: optional overrides per view.
    """
    id: str
    layer_id: str
    params: Dict[str, Any] = field(default_factory=dict)
    base_weight: float = 1.0
    view_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class View:
    """Stakeholder perspective over the same norm."""
    name: str
    # Optional metadata later (description, owner, etc.)


@dataclass
class Norm:
    """
    A process norm: constraints + views + optional metadata.
    """
    constraints: List[Constraint]
    views: List[View] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_view_names(self) -> List[str]:
        return [v.name for v in self.views]
