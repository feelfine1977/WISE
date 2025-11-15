# wise/io/log_loader.py

from typing import Union
import pandas as pd


def load_event_log(
    path_or_buffer: Union[str, bytes, "pd.DataFrame"],
    case_id_col: str,
    activity_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """
    Load a CSV event log OR normalise an already loaded DataFrame.

    Ensures:
    - required columns exist
    - timestamp column is converted to datetime
    """
    if isinstance(path_or_buffer, pd.DataFrame):
        df = path_or_buffer.copy()
    else:
        df = pd.read_csv(path_or_buffer)

    missing = {case_id_col, activity_col, timestamp_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    return df
