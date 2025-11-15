import pandas as pd
import pytest
from wise.model import Constraint, Norm, View


@pytest.fixture
def simple_log():
    data = [
        {"case_id": "A", "activity": "Create", "time": "2023-01-01"},
        {"case_id": "A", "activity": "Record GR", "time": "2023-01-02"},
        {"case_id": "A", "activity": "Record INV", "time": "2023-01-03"},
        {"case_id": "B", "activity": "Create", "time": "2023-01-01"},
    ]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    return df


@pytest.fixture
def presence_norm():
    c1 = Constraint(
        id="c1",
        layer_id="presence",
        params={"activity": "Record GR"},
        base_weight=1.0,
    )
    norm = Norm(constraints=[c1], views=[View(name="Default")])
    return norm
