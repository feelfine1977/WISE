from typing import List

import pandas as pd

from .eb import shrink_mean
from wise.model import Norm


def aggregate_slices(
    df_scores: pd.DataFrame,
    df_log: pd.DataFrame,
    case_id_col: str,
    slice_cols: List[str],
    shrink_k: float = 0.0,
) -> pd.DataFrame:
    """
    Aggregate case scores over slices and compute Priority Index (PI).

    Parameters
    ----------
    df_scores : DataFrame
        Output of compute_case_scores with columns [case_id_col, "score"].
    df_log : DataFrame
        Original event log (to get slice attributes).
    case_id_col : str
        Case identifier column.
    slice_cols : list of str
        Columns to define slices (e.g. ["case_Company", "case_Spend_area_text"]).
    shrink_k : float, default 0.0
        Empirical-Bayes shrinkage factor. 0 disables shrinkage.

    Returns
    -------
    DataFrame with:
      slice_cols + ["n_cases", "mean_score", "shrunk_mean_score", "gap", "PI"]
    """
    if not slice_cols:
        return pd.DataFrame()

    slice_df = df_log[[case_id_col] + slice_cols].drop_duplicates()
    merged = pd.merge(df_scores, slice_df, on=case_id_col, how="left")

    global_mean = merged["score"].mean()

    group = merged.groupby(slice_cols, dropna=False)
    agg = group["score"].agg(["count", "mean"]).reset_index()
    agg = agg.rename(columns={"count": "n_cases", "mean": "mean_score"})

    if shrink_k and shrink_k > 0:
        agg["shrunk_mean_score"] = shrink_mean(
            series=agg["mean_score"],
            global_mean=global_mean,
            counts=agg["n_cases"],
            k=shrink_k,
        )
        used_mean = "shrunk_mean_score"
    else:
        agg["shrunk_mean_score"] = agg["mean_score"]
        used_mean = "shrunk_mean_score"

    agg["gap"] = global_mean - agg[used_mean]
    agg["PI"] = agg["n_cases"] * agg["gap"]

    return agg.sort_values("PI", ascending=False)


def compute_slice_layer_matrix(
    df_scores: pd.DataFrame,
    df_log: pd.DataFrame,
    case_id_col: str,
    slice_cols: List[str],
) -> pd.DataFrame:
    """
    Build a slice × layer matrix with mean layer violations and gaps.

    Expects df_scores to have columns:
      - case_id_col
      - "score"
      - "violation_<layer_id>" for each layer (0..1).
    """
    if not slice_cols:
        return pd.DataFrame()

    layer_cols = [c for c in df_scores.columns if c.startswith("violation_")]
    if not layer_cols:
        return pd.DataFrame()

    slice_df = df_log[[case_id_col] + slice_cols].drop_duplicates()
    merged = pd.merge(df_scores, slice_df, on=case_id_col, how="left")

    global_means = {col: merged[col].mean() for col in layer_cols}

    group = merged.groupby(slice_cols, dropna=False)
    rows = []
    for s_key, g in group:
        if not isinstance(s_key, tuple):
            s_key = (s_key,)
        row = dict(zip(slice_cols, s_key))
        row["n_cases"] = len(g)
        for col in layer_cols:
            mean_v = g[col].mean()
            gap = mean_v - global_means[col]
            row[f"{col}_mean"] = mean_v
            row[f"{col}_gap"] = gap
        rows.append(row)

    return pd.DataFrame(rows)


def compute_slice_constraint_matrix(
    df_scores: pd.DataFrame,
    df_log: pd.DataFrame,
    case_id_col: str,
    slice_cols: List[str],
    norm: Norm,
    layer_id: str,
) -> pd.DataFrame:
    """
    Build a slice × constraint matrix for a single layer.

    Expects df_scores to have columns 'viol_<constraint_id>' for each constraint.
    Returns one row per slice with columns:
      slice_cols + ["n_cases", "<cid>_mean", "<cid>_gap", ...]
    """
    if not slice_cols:
        return pd.DataFrame()

    constraints = [c for c in norm.constraints if c.layer_id == layer_id]
    if not constraints:
        return pd.DataFrame()

    viol_cols = [f"viol_{c.id}" for c in constraints if f"viol_{c.id}" in df_scores.columns]
    if not viol_cols:
        return pd.DataFrame()

    slice_df = df_log[[case_id_col] + slice_cols].drop_duplicates()
    merged = pd.merge(df_scores, slice_df, on=case_id_col, how="left")

    global_means = {col: merged[col].mean() for col in viol_cols}

    group = merged.groupby(slice_cols, dropna=False)
    rows = []
    for s_key, g in group:
        if not isinstance(s_key, tuple):
            s_key = (s_key,)
        row = dict(zip(slice_cols, s_key))
        row["n_cases"] = len(g)
        for c in constraints:
            col = f"viol_{c.id}"
            if col not in g.columns:
                continue
            mean_v = g[col].mean()
            gap = mean_v - global_means[col]
            row[f"{c.id}_mean"] = mean_v
            row[f"{c.id}_gap"] = gap
        rows.append(row)

    return pd.DataFrame(rows)


def rank_slices(
    slice_summary: pd.DataFrame,
    min_cases: int = 50,
    top_n: int = 10,
    metric: str = "PI",
) -> pd.DataFrame:
    """
    Rank slices by a chosen metric ('PI', 'gap', 'PI_abs', 'gap_abs') and
    filter out slices with fewer than `min_cases`.

    Returns the top_n rows.
    """
    if slice_summary is None or slice_summary.empty:
        return pd.DataFrame()

    df = slice_summary.copy()
    df = df[df["n_cases"] >= min_cases]

    if metric == "PI_abs":
        df["PI_abs"] = df["PI"].abs()
        df = df.sort_values("PI_abs", ascending=False)
    elif metric == "gap_abs":
        df["gap_abs"] = df["gap"].abs()
        df = df.sort_values("gap_abs", ascending=False)
    elif metric == "gap":
        df = df.sort_values("gap", ascending=False)
    else:  # default PI
        df = df.sort_values("PI", ascending=False)

    return df.head(top_n)
