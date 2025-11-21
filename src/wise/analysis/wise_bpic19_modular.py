#!/usr/bin/env python
"""
WISE BPIC 2019 – Modular P2P Analysis Library

This module exposes reusable functions so you can:

- use the full pipeline as a script, or
- import individual steps from a Jupyter notebook, e.g.:

    from wise.analysis.wise_bpic19_modular import (
        load_settings_and_paths,
        load_log,
        subset_by_item_category,
        build_case_features,
        compute_wise_case_scores,
        enrich_case_full,
        aggregate_slices,
    )

    settings, paths = load_settings_and_paths()
    df = load_log(paths["data_path"], settings)
    df_example = subset_by_item_category(df, "3-way match, invoice after GR",
                                         category_col="case Item Category",
                                         case_id_col=settings["CASE_COLS"]["CASE_ID_COL"])

    feat = build_case_features(df_example, case_id_col, act_col, ts_col)
    case_scores = compute_wise_case_scores(df_example, paths["norm_path"],
                                           view_name="Finance",
                                           case_id_col=case_id_col,
                                           act_col=act_col,
                                           ts_col=ts_col)

    case_full = enrich_case_full(case_scores, feat, df_example, case_id_col)
    slice_summary = aggregate_slices(case_full,
                                     slice_cols=["case Company", "case Spend area text", "complexity_cluster"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.filtering.log.variants import variants_filter

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# WISE-specific imports (assumes WISE is on PYTHONPATH)
from wise.io.norm_loader import load_norm_from_json
from wise.scoring.scoring import compute_case_scores


# ============================================================================
# 0. SETTINGS / PATHS
# ============================================================================

def load_settings(settings_path: Path) -> dict:
    """Load settings.json from a given path."""
    with settings_path.open("r", encoding="utf-8") as f:
        settings = json.load(f)
    return settings


def default_settings_path() -> Path:
    """
    Compute a default settings.json path assuming this file lives at:
      <project-root>/WISE/Untitled/src/wise/analysis/this_file.py

    Adjust if your layout differs.
    """
    here = Path(__file__).resolve()
    return here.parent.parent.parent.parent / "settings.json"


def load_settings_and_paths(
    settings_path: Optional[Path] = None,
) -> Tuple[dict, Dict[str, Path]]:
    """
    Load settings.json and resolve data/norm/output paths.

    Returns
    -------
    settings : dict
    paths : dict with keys "settings_path", "data_path", "norm_path", "output_path"
    """
    if settings_path is None:
        settings_path = default_settings_path()

    settings = load_settings(settings_path)
    base = settings_path.parent

    data_path = base / settings["data_path"]
    norm_path = base / settings["norm_path"]
    output_path = base / settings.get("output_path", "output")
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "settings_path": settings_path,
        "data_path": data_path,
        "norm_path": norm_path,
        "output_path": output_path,
    }
    return settings, paths


# ============================================================================
# 1. LOG LOADING / SUBSETTING
# ============================================================================

def load_log(
    data_path: Path,
    settings: dict,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load the BPIC log (or any CSV) according to settings.

    Parameters
    ----------
    data_path : Path
        CSV file path.
    settings : dict
        Settings that contain CASE_COLS with TIMESTAMP_COL.
    parse_dates : bool
        Whether to parse the timestamp column as datetime.

    Returns
    -------
    df : pandas.DataFrame
    """
    df = pd.read_csv(data_path, encoding="latin1", low_memory=False)

    if parse_dates:
        ts_col = settings["CASE_COLS"]["TIMESTAMP_COL"]
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    case_id_col = settings["CASE_COLS"]["CASE_ID_COL"]
    n_events = len(df)
    n_cases = df[case_id_col].nunique()
    print(f"[INFO] Loaded log from {data_path}: {n_events:,} events, {n_cases:,} cases")

    return df


def subset_by_item_category(
    df: pd.DataFrame,
    cat_value: str,
    category_col: str,
    case_id_col: str,
) -> pd.DataFrame:
    """
    Restrict df to all cases that have case[category_col] == cat_value.

    Can be called from notebooks with any log, not just BPIC.

    Parameters
    ----------
    df : DataFrame
    cat_value : str
        Value of the category to filter on.
    category_col : str
        Case-level category column (e.g. "case Item Category").
    case_id_col : str

    Returns
    -------
    subset : DataFrame
    """
    cases = df.loc[df[category_col] == cat_value, case_id_col].unique()
    subset = df[df[case_id_col].isin(cases)].copy()
    print(f"[INFO] Subset '{cat_value}': {len(subset):,} events, {subset[case_id_col].nunique():,} cases")
    return subset


# ============================================================================
# 2. OPTIONAL: PM4PY VISUALISATIONS (NOT USED IN BATCH PIPELINES)
# ============================================================================

def pm4py_log(df_subset: pd.DataFrame, case_id_col: str, act_col: str, ts_col: str) -> pd.DataFrame:
    """Convert a subset DataFrame into a pm4py-compatible event log DataFrame."""
    df_pm = df_subset[[case_id_col, act_col, ts_col]].rename(
        columns={
            case_id_col: case_id_col,
            act_col: act_col,
            ts_col: ts_col,
        }
    )
    df_pm = pm4py.format_dataframe(
        df_pm,
        case_id=case_id_col,
        activity_key=act_col,
        timestamp_key=ts_col,
    )
    return df_pm




def show_for_category(
    df_subset: pd.DataFrame,
    category_name: str,
    case_id_col: str,
    act_col: str,
    ts_col: str,
    percentile: float = 0.9,
) -> None:
    """
    Show DFG + Petri net for a given subset.

    percentile ∈ (0, 1]:
      - DFG: keep most frequent edges until the cumulative frequency
        reaches that percentile of all edge frequencies.
      - Petri net: keep most frequent variants until cumulative trace
        count reaches that percentile of all traces.

    Call this only in interactive use (Jupyter), not in batch.
    """
    print(f"\n[PROCESS MODEL] Category: {category_name}")
    n_events = len(df_subset)
    n_cases = df_subset[case_id_col].nunique()
    print(f"Events: {n_events:,}, Cases: {n_cases:,}")

    # ------------------------------------------------------------------
    # Prepare pm4py log
    # ------------------------------------------------------------------
    df_pm = pm4py_log(df_subset, case_id_col, act_col, ts_col)

    # ------------------------------------------------------------------
    # 1) DFG: filter edges by frequency percentile
    # ------------------------------------------------------------------
    dfg, starts, ends = pm4py.discover_dfg(df_pm)

    dfg_to_view = dfg
    if 0 < percentile < 1 and dfg:
        total_freq = sum(dfg.values())
        if total_freq > 0:
            # sort edges by frequency, descending
            sorted_edges = sorted(dfg.items(), key=lambda kv: kv[1], reverse=True)

            cum = 0
            dfg_filtered = {}
            for edge, freq in sorted_edges:
                dfg_filtered[edge] = freq
                cum += freq
                if cum / total_freq >= percentile:
                    break
            dfg_to_view = dfg_filtered

    if dfg_to_view:
        pm4py.view_dfg(dfg_to_view, starts, ends)
    else:
        print("[INFO] DFG is empty after filtering, skipping DFG view.")

    # ------------------------------------------------------------------
    # 2) Petri net: filter log variants by frequency percentile
    # ------------------------------------------------------------------
    log = pm4py.convert_to_event_log(df_pm)

    log_filtered = log
    if 0 < percentile < 1:
        # pm4py.get_variants is available in your version (you used it before)
        variants = pm4py.get_variants(log)
        if variants:
            total_traces = sum(len(traces) for traces in variants.values())
            if total_traces > 0:
                # sort variants by frequency, descending
                sorted_vars = sorted(
                    variants.items(),
                    key=lambda kv: len(kv[1]),
                    reverse=True,
                )

                cum = 0
                selected_variant_keys = []
                for var_key, traces in sorted_vars:
                    selected_variant_keys.append(var_key)
                    cum += len(traces)
                    if cum / total_traces >= percentile:
                        break

                # use variants_filter.apply instead of non-existent pm4py.filter_log_variants
                log_filtered = variants_filter.apply(log, selected_variant_keys)

    if len(log_filtered) == 0:
        print("[INFO] Filtered log is empty, skipping Petri net discovery.")
        return

    net, im, fm = pm4py.discover_petri_net_inductive(log_filtered)
    pm4py.view_petri_net(net, im, fm)





# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

def build_case_features(
    df_example: pd.DataFrame,
    case_id_col: str,
    act_col: str,
    ts_col: str,
) -> pd.DataFrame:
    """
    Build case-level features: lead times, counts, start-of-case, automation, cluster.

    Parameters
    ----------
    df_example : DataFrame
        Event log subset (e.g. one item category).
    case_id_col : str
    act_col : str
    ts_col : str

    Returns
    -------
    feat : DataFrame
        One row per case, indexed by case_id_col.
    """
    ACT_PO_CREATE = "Create Purchase Order Item"
    ACT_GR = "Record Goods Receipt"
    ACT_INV = "Record Invoice Receipt"
    ACT_CLEAR = "Clear Invoice"

    def first_occurrence(df_subset: pd.DataFrame, act: str) -> pd.Series:
        sub = df_subset[df_subset[act_col] == act]
        return sub.groupby(case_id_col)[ts_col].min()

    first_po = first_occurrence(df_example, ACT_PO_CREATE)
    first_gr = first_occurrence(df_example, ACT_GR)
    first_inv = first_occurrence(df_example, ACT_INV)
    first_clear = first_occurrence(df_example, ACT_CLEAR)

    feat = pd.DataFrame(
        {
            "first_po": first_po,
            "first_gr": first_gr,
            "first_inv": first_inv,
            "first_clear": first_clear,
        }
    )

    # Lead times
    feat["po_to_gr_days"] = (feat["first_gr"] - feat["first_po"]).dt.days
    feat["gr_to_inv_days"] = (feat["first_inv"] - feat["first_gr"]).dt.days
    feat["inv_to_clear_days"] = (feat["first_clear"] - feat["first_inv"]).dt.days

    # Counts
    sub = df_example.copy()
    sub["is_gr"] = sub[act_col].eq(ACT_GR).astype(int)
    sub["is_inv"] = sub[act_col].eq(ACT_INV).astype(int)
    sub["is_clear"] = sub[act_col].eq(ACT_CLEAR).astype(int)
    sub["is_price_change"] = sub[act_col].str.contains("Change Price", na=False).astype(int)

    resource_col = None
    for cand in ["event org:resource", "org:resource"]:
        if cand in sub.columns:
            resource_col = cand
            break

    agg_dict = {
        "n_events": (act_col, "size"),
        "n_gr": ("is_gr", "sum"),
        "n_inv": ("is_inv", "sum"),
        "n_clear": ("is_clear", "sum"),
        "n_price_changes": ("is_price_change", "sum"),
    }
    if resource_col is not None:
        agg_dict["n_users"] = (resource_col, "nunique")

    counts = sub.groupby(case_id_col).agg(**agg_dict).reset_index()

    feat = feat.reset_index().rename(columns={"index": case_id_col})
    feat = feat.merge(counts, on=case_id_col, how="left")

    if "n_users" not in feat.columns:
        feat["n_users"] = np.nan

    # Start-of-case ts
    first_case_ts = df_example.groupby(case_id_col)[ts_col].min()
    feat = feat.merge(
        first_case_ts.rename("start_ts").reset_index(),
        on=case_id_col,
        how="left",
    )
    feat["start_dow"] = feat["start_ts"].dt.dayofweek
    feat["start_month"] = feat["start_ts"].dt.month

    # Automation heuristic
    if resource_col is not None:
        users = df_example[resource_col].value_counts()
        batch_users = {u for u in users.index if isinstance(u, str) and u.startswith("batch_user")}

        df_example = df_example.copy()
        df_example["is_batch_user"] = df_example[resource_col].isin(batch_users)

        auto_counts = df_example.groupby(case_id_col)["is_batch_user"].agg(
            n_auto_events=lambda x: x.sum(),
            n_manual_events=lambda x: (~x).sum(),
        )
        auto_counts["auto_ratio"] = auto_counts["n_auto_events"] / (
            auto_counts["n_auto_events"] + auto_counts["n_manual_events"]
        )

        feat = feat.merge(auto_counts.reset_index(), on=case_id_col, how="left")
    else:
        feat["n_auto_events"] = np.nan
        feat["n_manual_events"] = np.nan
        feat["auto_ratio"] = np.nan

    #add to feature if is_gr, is_inv, is_clear columns as True/False
    feat["has_gr"] = feat["n_gr"] > 0
    feat["has_inv"] = feat["n_inv"] > 0
    feat["has_clear"] = feat["n_clear"] > 0

    # ------------------------------------------------------------------
    # multiplicity features: repeated activities & repeated 'change' acts
    # ------------------------------------------------------------------

    # count events per (case, activity)
    case_act_counts = (
        df_example
        .groupby([case_id_col, act_col])
        .size()
        .rename("act_count")
        .reset_index()
    )

    # ---- 1) All activities with multiple occurrences in a case --------
    multi = case_act_counts[case_act_counts["act_count"] > 1]

    multi_per_case = (
        multi.groupby(case_id_col)["act_count"]
        .agg(
            n_multi_activities="size",   # how many activity types are repeated
            sum_multi_events="sum",      # total events in repeated activities
        )
        .reset_index()
    )

    # ---- 2) Only 'change' activities with multiple occurrences --------
    change_mask = case_act_counts[act_col].str.contains("change", case=False, na=False)
    multi_change = case_act_counts[change_mask & (case_act_counts["act_count"] > 1)]

    multi_change_per_case = (
        multi_change.groupby(case_id_col)["act_count"]
        .agg(
            n_multi_change_activities="size",
            sum_multi_change_events="sum",
        )
        .reset_index()
    )

    # ---- 3) Merge into feat and fill missing with 0 -------------------
    feat = feat.merge(multi_per_case, on=case_id_col, how="left")
    feat = feat.merge(multi_change_per_case, on=case_id_col, how="left")

    for col in [
        "n_multi_activities",
        "sum_multi_events",
        "n_multi_change_activities",
        "sum_multi_change_events",
    ]:
        feat[col] = feat[col].fillna(0).astype(int)

    # KMeans cluster
    feature_cols = [
        "po_to_gr_days",
        "gr_to_inv_days",
        "inv_to_clear_days",
        "n_events",
        "n_gr",
        "n_inv",
        "n_clear",
        "n_price_changes",
        "n_users",
        "auto_ratio",
        "n_multi_activities",
        "n_multi_change_activities",
    ]
    X = feat[feature_cols].fillna(0.0).values
    preproc = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=4, random_state=42)),
        ]
    )
    feat["cluster"] = preproc.fit_predict(X)

    


    
    return feat


# ============================================================================
# 4. WISE SCORING
# ============================================================================

def compute_wise_case_scores(
    df_example: pd.DataFrame,
    norm_path: Path,
    view_name: str,
    case_id_col: str,
    act_col: str,
    ts_col: str,
) -> pd.DataFrame:
    """
    Compute WISE case scores for a given event log subset.

    You can call this from any notebook with your own df_example.

    Returns
    -------
    case_scores : DataFrame with columns [case_id_col, 'score', 'badness', ...]
    """
    with norm_path.open("r", encoding="utf-8") as f:
        norm = load_norm_from_json(f)

    case_scores = compute_case_scores(
        df=df_example,
        norm=norm,
        view_name=view_name,
        case_id_col=case_id_col,
        activity_col=act_col,
        timestamp_col=ts_col,
    )
    case_scores["badness"] = 1.0 - case_scores["score"]
    return case_scores


# ============================================================================
# 5. CASE_FULL ENRICHMENT & CLUSTERS
# ============================================================================

def make_speed_clusters(df: pd.DataFrame, src_col: str, out_col: str, q: int = 5) -> pd.DataFrame:
    """
    Cluster src_col into up to q quantile-based bins.
    Label: _1 = slowest (largest value), _N = fastest (smallest value).
    """
    cats = pd.qcut(df[src_col], q, duplicates="drop")
    n = cats.cat.categories.size
    new_labels = {old_cat: f"_{i}" for i, old_cat in zip(range(n, 0, -1), cats.cat.categories)}
    df[out_col] = cats.cat.rename_categories(new_labels)
    return df


def enrich_case_full(
    case_scores: pd.DataFrame,
    feat: pd.DataFrame,
    df_example: pd.DataFrame,
    case_id_col: str,
) -> pd.DataFrame:
    """
    Combine scores, engineered features, and slice attributes into case_full.

    This function is fully reusable with any norms / features you build.

    Returns
    -------
    case_full : DataFrame
        One row per case containing score, badness, features, and slice attributes.
    """
    case_full = case_scores.merge(feat, on=case_id_col, how="left")

    # Attach categorical slices from original df_example
    slice_cols = [
        "case Company",
        "case Spend area text",
        "case Item Category",
        "case Source",
    ]
    existing_slice_cols = [c for c in slice_cols if c in df_example.columns]

    if existing_slice_cols:
        slice_attrs = df_example[[case_id_col] + existing_slice_cols].drop_duplicates()
        case_full = case_full.merge(slice_attrs, on=case_id_col, how="left")

    # Lead-time clusters
    for col, out in [
        ("po_to_gr_days", "po_to_gr_cluster"),
        ("gr_to_inv_days", "gr_to_inv_cluster"),
        ("inv_to_clear_days", "inv_to_clear_cluster"),
    ]:
        if col in case_full.columns:
            case_full = make_speed_clusters(case_full, col, out, q=5)

    # Automation level: _1 = most manual, _N = most automated
    if "auto_ratio" in case_full.columns:
        cats = pd.qcut(case_full["auto_ratio"], 5, duplicates="drop")
        n_bins = cats.cat.categories.size
        labels = [f"_{i}" for i in range(1, n_bins + 1)]
        mapping = {old: lab for old, lab in zip(cats.cat.categories, labels)}
        case_full["auto_level"] = cats.cat.rename_categories(mapping)

    # Complexity score + clusters
    complexity_features = ["n_events", "n_gr", "n_inv", "n_price_changes", "n_clear", "n_multi_activities", "n_multi_change_activities", "n_users"]
    existing_complexity = [c for c in complexity_features if c in case_full.columns]
    if existing_complexity:
        ranked = case_full[existing_complexity].rank(pct=True)
        case_full["complexity_score"] = ranked.mean(axis=1)

        case_full["complexity_cluster"] = pd.qcut(
            case_full["complexity_score"],
            4,
            labels=[f"_{i}" for i in range(1, 5)],
            duplicates="drop",
        )

    # Throughput time & cluster
    if {"po_to_gr_days", "gr_to_inv_days", "inv_to_clear_days"}.issubset(case_full.columns):
        case_full["throughput_days"] = (
            case_full["inv_to_clear_days"]
            + case_full["po_to_gr_days"]
            + case_full["gr_to_inv_days"]
        )
        case_full["throughput_cluster"] = pd.qcut(
            case_full["throughput_days"],
            5,
            labels=[f"_{i}" for i in range(5, 0, -1)],  # _1 slowest, _5 fastest
            duplicates="drop",
        )

    # Season & weekday/weekend & dayofweek
    if "start_month" in case_full.columns:
        case_full["start_season"] = pd.cut(
            case_full["start_month"],
            bins=[0, 3, 6, 9, 12],
            labels=["Q1", "Q2", "Q3", "Q4"],
            include_lowest=True,
        )
    if "start_dow" in case_full.columns:
        case_full["start_weekpart"] = np.where(
            case_full["start_dow"].isin([5, 6]),
            "weekend",
            "weekday",
        )
    if "start_dow" in case_full.columns:
        case_full["start_dayofweek"] = case_full["start_dow"].apply(
            lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x]
        )

    return case_full


# ============================================================================
# 6. SLICE AGGREGATION
# ============================================================================

def aggregate_slices(
    df_scores: pd.DataFrame,
    slice_cols: List[str],
    score_col: str = "score",
    shrink_k: float = 50.0,
) -> pd.DataFrame:
    """
    Aggregate scores over slices with empirical-Bayes shrinkage.

    Parameters
    ----------
    df_scores : DataFrame
        Must contain score_col and slice_cols.
    slice_cols : list of str
        Columns to group by.
    score_col : str
        Score column name.
    shrink_k : float
        Shrinkage strength.

    Returns
    -------
    agg : DataFrame
        Contains slice_cols, n_cases, mean_score, shrunk_score.
    """
    df = df_scores.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    group = df.groupby(slice_cols, dropna=False)
    agg = group[score_col].agg(["count", "mean"]).reset_index()
    agg = agg.rename(columns={"count": "n_cases", "mean": "mean_score"})

    valid = agg["mean_score"].notna() & (agg["n_cases"] > 0)
    if valid.any():
        global_mean = np.average(
            agg.loc[valid, "mean_score"],
            weights=agg.loc[valid, "n_cases"],
        )
    else:
        global_mean = df[score_col].mean()

    if np.isnan(global_mean):
        agg["shrunk_score"] = agg["mean_score"]
    else:
        agg["shrunk_score"] = (
            agg["n_cases"] * agg["mean_score"] + shrink_k * global_mean
        ) / (agg["n_cases"] + shrink_k)

    return agg


# ============================================================================
# 7. FULL PIPELINE (OPTIONAL SCRIPT ENTRYPOINT)
# ============================================================================

def run_full_pipeline(
    settings_path: Optional[Path] = None,
    view_name: str = "Finance",
    process_type_text: str = "3-way match, invoice after GR",
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the whole BPIC19 pipeline:

    - load settings & log
    - subset by item category
    - build features
    - compute WISE scores
    - build case_full
    - aggregate slices

    Returns
    -------
    case_scores, case_full, slice_summary
    """
    # Settings & paths
    settings, paths = load_settings_and_paths(settings_path)
    data_path = paths["data_path"]
    norm_path = paths["norm_path"]
    output_path = paths["output_path"]

    case_id_col = settings["CASE_COLS"]["CASE_ID_COL"]
    act_col = settings["CASE_COLS"]["ACTIVITY_COL"]
    ts_col = settings["CASE_COLS"]["TIMESTAMP_COL"]

    # Log
    df = load_log(data_path, settings, parse_dates=True)

    # Subset
    category_col = "case Item Category"
    if category_col not in df.columns:
        raise KeyError(f"{category_col!r} not in log columns")

    df_example = subset_by_item_category(
        df=df,
        cat_value=process_type_text,
        category_col=category_col,
        case_id_col=case_id_col,
    )

    # Features & scores
    feat = build_case_features(df_example, case_id_col, act_col, ts_col)
    case_scores = compute_wise_case_scores(
        df_example=df_example,
        norm_path=norm_path,
        view_name=view_name,
        case_id_col=case_id_col,
        act_col=act_col,
        ts_col=ts_col,
    )

    # Case_full
    case_full = enrich_case_full(case_scores, feat, df_example, case_id_col)

    # Slice aggregation
    slice_cols = ["case Company", "case Spend area text"] + [
        col for col in case_full.columns if "cluster" in col
    ]
    slice_cols = [c for c in slice_cols if c in case_full.columns]
    slice_summary = aggregate_slices(case_full, slice_cols, score_col="score", shrink_k=50.0)
    slice_summary = slice_summary.sort_values("mean_score")

    if save_outputs:
        output_path_scores = output_path / "bpic19_case_scores.csv"
        output_path_case_full = output_path / "bpic19_case_full.csv"
        output_path_slice_summary = output_path / "bpic19_slice_summary.csv"

        case_scores.to_csv(output_path_scores, index=False)
        case_full.to_csv(output_path_case_full, index=False)
        slice_summary.to_csv(output_path_slice_summary, index=False)

        print(f"[INFO] Saved case scores to {output_path_scores}")
        print(f"[INFO] Saved case_full to {output_path_case_full}")
        print(f"[INFO] Saved slice summary to {output_path_slice_summary}")

    return case_scores, case_full, slice_summary


if __name__ == "__main__":
    # Script entrypoint: run full pipeline and save outputs
    run_full_pipeline()
