from typing import List
import pandas as pd

from .eb import shrink_mean


def aggregate_slices(
    df_scores: pd.DataFrame,
    df_log: pd.DataFrame,
    case_id_col: str,
    slice_cols: List[str],
    shrink_k: float = 0.0,
) -> pd.DataFrame:
    """
    Join case scores with slice attributes and compute PI per slice.
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
        used_mean = "mean_score"

    agg["gap"] = global_mean - agg[used_mean]
    agg["PI"] = agg["n_cases"] * agg["gap"]

    return agg.sort_values("PI", ascending=False)
