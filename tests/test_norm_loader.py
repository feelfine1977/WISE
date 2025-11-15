import io
from wise.io.norm_loader import load_norm_from_json


def test_load_norm_from_json_basic():
    raw = io.StringIO(
        """
        {
          "views": ["Finance"],
          "constraints": [
            {
              "id": "c1",
              "layer_id": "presence",
              "params": {"activity": "Record GR"},
              "base_weight": 1.0,
              "view_weights": {"Finance": 0.5}
            }
          ]
        }
        """
    )
    norm = load_norm_from_json(raw)
    assert len(norm.constraints) == 1
    assert norm.constraints[0].layer_id == "presence"
    assert norm.get_view_names() == ["Finance"]
