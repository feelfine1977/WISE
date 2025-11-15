from wise.layers import get_layer
from wise.model import Constraint
import pandas as pd


def test_presence_layer_violation():
    layer = get_layer("presence")
    df = pd.DataFrame(
        [
            {"act": "A", "time": "2023-01-01"},
            {"act": "B", "time": "2023-01-02"},
        ]
    )
    df["time"] = pd.to_datetime(df["time"])

    c = Constraint(id="c1", layer_id="presence", params={"activity": "B"})
    v = layer.compute_violation(df, c, activity_col="act", timestamp_col="time")
    assert v == 0.0

    c_missing = Constraint(id="c2", layer_id="presence", params={"activity": "C"})
    v_missing = layer.compute_violation(df, c_missing, "act", "time")
    assert v_missing == 1.0
