from typing import Dict, Type

from .base import BaseLayer
from .presence import PresenceLayer
from .order_lag import OrderLagLayer
from .balance import BalanceLayer
from .singularity import SingularityLayer
from .exclusion import ExclusionLayer

# Registry: layer_id -> layer class
_LAYER_REGISTRY: Dict[str, Type[BaseLayer]] = {}


def register_layer(layer_cls: Type[BaseLayer]) -> Type[BaseLayer]:
    """Decorator to register a new layer implementation."""
    layer_id = getattr(layer_cls, "LAYER_ID", None)
    if not layer_id:
        raise ValueError(f"Layer class {layer_cls.__name__} has no LAYER_ID")
    _LAYER_REGISTRY[layer_id] = layer_cls
    return layer_cls


def get_layer(layer_id: str) -> BaseLayer:
    """Instantiate a layer implementation by id."""
    if layer_id not in _LAYER_REGISTRY:
        raise KeyError(f"Unknown layer_id '{layer_id}'. Registered: {list(_LAYER_REGISTRY.keys())}")
    return _LAYER_REGISTRY[layer_id]()


# Register built-in layers
register_layer(PresenceLayer)
register_layer(OrderLagLayer)
register_layer(BalanceLayer)
register_layer(SingularityLayer)
register_layer(ExclusionLayer)
