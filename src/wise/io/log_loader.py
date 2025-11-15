from typing import Union
import pandas as pd


def load_event_log(
    path_or_buffer: Union[str, bytes],
    case_id_col: str,
    activity_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """Load a CSV event log and ensure the minimum columns exist."""
    df = pd.read_csv(path_or_buffer)
    missing = {case_id_col, activity_col, timestamp_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    return df
