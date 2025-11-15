from typing import Union

import pandas as pd


def load_event_log(
    path_or_buffer: Union[str, bytes, "pd.DataFrame"],
    case_id_col: str,
    activity_col: str,
    timestamp_col: str,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load a CSV event log OR normalise an already loaded DataFrame.

    Ensures:
    - required columns exist
    - timestamp column is converted to datetime

    Parameters
    ----------
    path_or_buffer : str | bytes | DataFrame
        Path to a CSV file or an already loaded DataFrame.
    case_id_col : str
        Column name for the case identifier.
    activity_col : str
        Column name for the activity label.
    timestamp_col : str
        Column name for the event timestamp.

    Returns
    -------
    DataFrame
        Normalised event log.
    """
    if isinstance(path_or_buffer, pd.DataFrame):
        df = path_or_buffer.copy()
    else:
        df = pd.read_csv(path_or_buffer, encoding=encoding, low_memory=False)

    missing = {case_id_col, activity_col, timestamp_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    return df
