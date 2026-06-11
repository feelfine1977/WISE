"""Generic case-level features from the canonical event frame."""
import numpy as np
import pandas as pd

from .mapping import slug


def build_case_features(events: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """One row per case: activity counts, first timestamps, generic stats,
    pass-through dimensions, numeric-attribute totals."""
    g = events.groupby("case_id", sort=False)

    base = g.agg(total_events=("activity", "size"),
                 case_first_ts=("timestamp", "min"),
                 case_last_ts=("timestamp", "max"),
                 distinct_resources=("resource", "nunique"),
                 distinct_timestamps=("timestamp", "nunique")).reset_index()
    base["event_replication_ratio"] = base["total_events"] / base["distinct_timestamps"].clip(lower=1)
    base["case_duration_days"] = ((base["case_last_ts"] - base["case_first_ts"])
                                  .dt.total_seconds() / 86400)
    base["case_start_quarter"] = base["case_first_ts"].dt.to_period("Q").astype(str)

    # activity counts + first occurrence per activity (pivot once)
    cnt = (events.pivot_table(index="case_id", columns="activity",
                              values="timestamp", aggfunc="size", fill_value=0))
    cnt.columns = [f"count__{slug(a)}" for a in cnt.columns]
    first = (events.pivot_table(index="case_id", columns="activity",
                                values="timestamp", aggfunc="min"))
    first.columns = [f"first_ts__{slug(a)}" for a in first.columns]

    out = base.merge(cnt, on="case_id").merge(first, on="case_id", how="left")

    # dimensions: first value per case
    dims = [c for c in mapping.get("dimensions", []) if c in events.columns]
    if dims:
        out = out.merge(g[dims].first().reset_index(), on="case_id")
    if "document_id" in events.columns:
        out = out.merge(g["document_id"].first().reset_index(), on="case_id")

    # numeric attributes: per-case total + max
    for a in mapping.get("numeric_attributes", []):
        if a in events.columns:
            agg = g[a].agg(["sum", "max"])
            out[f"total__{slug(a)}"] = out["case_id"].map(agg["sum"])
            out[f"max__{slug(a)}"] = out["case_id"].map(agg["max"])
    return out
