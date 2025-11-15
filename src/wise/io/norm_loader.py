from typing import IO, Dict, Any, List
import json

from ..model import Norm, Constraint, View


def load_norm_from_json(f: IO[str]) -> Norm:
    """
    Expected schema:

    {
      "views": ["Finance", "Logistics"],
      "constraints": [
        {
          "id": "c1",
          "layer_id": "presence",
          "params": {"activity": "Record Goods Receipt"},
          "base_weight": 1.0,
          "view_weights": {"Finance": 0.4, "Logistics": 0.2}
        },
        ...
      ]
    }
    """
    raw: Dict[str, Any] = json.load(f)
    views = [View(name=v) for v in raw.get("views", [])]
    constraints: List[Constraint] = []
    for c in raw.get("constraints", []):
        constraints.append(
            Constraint(
                id=c["id"],
                layer_id=c["layer_id"],
                params=c.get("params", {}),
                base_weight=c.get("base_weight", 1.0),
                view_weights=c.get("view_weights", {}),
            )
        )
    return Norm(constraints=constraints, views=views)
