"""
wise.features — per-case feature extraction (one row per PO item).

Constraints in WISE read *named, documented columns* rather than opaque
embeddings (paper auditability requirement): activity counts cnt(a,σ),
first-occurrence timestamps t1(a,σ), manual-touch statistics, and an
exposure proxy exp(σ) for the exposure-weighted PI variant.
"""
import re

import numpy as np
import pandas as pd


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def build_case_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the event log to one feature row per case σ.

    Produces, per paper Sec. IV-A:
      * cnt(a, σ)  → columns  count__<activity>
      * t1(a, σ)   → columns  first_ts__<activity>
      * exp(σ)     → exposure_abs_max / exposure_norm  (Cumulative net worth)
    plus effort features used by L7 constraints (manual_share, total_events,
    distinct_human_resources) and a net-worth volatility proxy used by the
    L6 balance-proxy constraints (networth_cv).
    """
    case_cols = [
        "case_id", "purchasing_document", "flow_type", "flow_type_raw",
        "case Company", "case Spend area text", "case Sub spend area text",
        "case Vendor", "case Item Type", "case Document Type",
        "case Purch. Doc. Category name", "case Item Category",
        "case Spend classification text", "case Source", "case Name",
        "case GR-Based Inv. Verif.", "case Item", "case Goods Receipt",
        "case_start_quarter",
    ]
    case_cols = [c for c in case_cols if c in df.columns]
    case_base = df.groupby("case_id", as_index=False)[case_cols[1:]].first()

    total_events = df.groupby("case_id").size().rename("total_events")
    manual_touch_count = df.groupby("case_id")["is_human_resource"].sum().rename("manual_touch_count")
    distinct_resources = df.groupby("case_id")["event org:resource"].nunique(dropna=True).rename("distinct_resources")
    human_tmp = df.loc[df["is_human_resource"], ["case_id", "event org:resource"]].dropna()
    distinct_human_resources = (
        human_tmp.groupby("case_id")["event org:resource"].nunique().rename("distinct_human_resources")
    )
    networth_abs = df["event Cumulative net worth (EUR)"].astype(float).abs()
    exposure_abs_max = networth_abs.groupby(df["case_id"]).max().rename("exposure_abs_max")
    networth = df["event Cumulative net worth (EUR)"].astype(float)
    grp = networth.groupby(df["case_id"])
    networth_cv = (grp.std().fillna(0.0) / grp.mean().clip(lower=1e-6)).rename("networth_cv")

    activity_count = df.groupby(["case_id", "event_activity"]).size().unstack(fill_value=0)
    activity_count.columns = [f"count__{slugify(c)}" for c in activity_count.columns]
    first_ts = df.groupby(["case_id", "event_activity"])["event time:timestamp"].min().unstack()
    first_ts.columns = [f"first_ts__{slugify(c)}" for c in first_ts.columns]

    # last event timestamp per case → needed by the governance loop
    # (right-censoring: "still active within 60 days of the window end")
    last_ts = df.groupby("case_id")["event time:timestamp"].max().rename("case_last_ts")
    first_any = df.groupby("case_id")["event time:timestamp"].min().rename("case_first_ts")
    # events per distinct timestamp → replication ratio (governance loop)
    distinct_ts = df.groupby("case_id")["event time:timestamp"].nunique().rename("distinct_timestamps")

    cf = case_base.merge(total_events.reset_index(), on="case_id", how="left")
    for s in [manual_touch_count, distinct_resources, distinct_human_resources,
              exposure_abs_max, networth_cv, last_ts, first_any, distinct_ts]:
        cf = cf.merge(s.reset_index(), on="case_id", how="left")
    cf = cf.merge(activity_count.reset_index(), on="case_id", how="left")
    cf = cf.merge(first_ts.reset_index(), on="case_id", how="left")

    for col in ["manual_touch_count", "distinct_resources", "distinct_human_resources", "networth_cv"]:
        cf[col] = cf[col].fillna(0).astype(float)
    cf["manual_share"] = np.where(cf["total_events"] > 0,
                                  cf["manual_touch_count"] / cf["total_events"], 0.0)
    exp = cf["exposure_abs_max"].fillna(0.0)
    q95 = float(exp.quantile(0.95)) or 1.0
    cf["exposure_norm"] = np.clip(exp / q95, 0, 1)
    cf["event_replication_ratio"] = (
        cf["total_events"] / cf["distinct_timestamps"].clip(lower=1)
    )
    return cf
