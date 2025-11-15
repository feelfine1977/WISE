"""
Helpers for managing WISE state in a Streamlit app.

We store three main objects in st.session_state:

- wise_dataset: information about the uploaded log and column mapping
- wise_norm: the current Norm object
- wise_results: computed case scores and slice summary
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import pandas as pd
import streamlit as st

from wise.model import Norm


# ---------------------------
# Dataset state
# ---------------------------

@dataclass
class DatasetState:
    df: pd.DataFrame
    case_id_col: str
    activity_col: str
    timestamp_col: str
    slice_cols: List[str]


def set_dataset_state(
    df: pd.DataFrame,
    case_id_col: str,
    activity_col: str,
    timestamp_col: str,
    slice_cols: Optional[List[str]] = None,
) -> None:
    if slice_cols is None:
        slice_cols = []
    st.session_state["wise_dataset"] = DatasetState(
        df=df,
        case_id_col=case_id_col,
        activity_col=activity_col,
        timestamp_col=timestamp_col,
        slice_cols=slice_cols,
    )


def get_dataset_state() -> Optional[DatasetState]:
    return st.session_state.get("wise_dataset")


# ---------------------------
# Norm state
# ---------------------------

def set_norm_state(norm: Norm) -> None:
    st.session_state["wise_norm"] = norm


def get_norm_state() -> Optional[Norm]:
    return st.session_state.get("wise_norm")


# ---------------------------
# Results state
# ---------------------------

@dataclass
class ResultsState:
    view_name: str
    case_scores: pd.DataFrame
    slice_summary: pd.DataFrame
    params: Dict[str, Any]


def set_results_state(
    view_name: str,
    case_scores: pd.DataFrame,
    slice_summary: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    if params is None:
        params = {}
    st.session_state["wise_results"] = ResultsState(
        view_name=view_name,
        case_scores=case_scores,
        slice_summary=slice_summary,
        params=params,
    )


def get_results_state() -> Optional[ResultsState]:
    return st.session_state.get("wise_results")


# ---------------------------
# Convenience
# ---------------------------

def reset_all() -> None:
    """Clear all WISE-related objects from session state."""
    for key in ["wise_dataset", "wise_norm", "wise_results"]:
        st.session_state.pop(key, None)
