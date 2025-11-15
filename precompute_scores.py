"""
Precompute WISE scores for a large BPIC 2019 log.

- Loads the raw CSV event log.
- Loads a WISE norm JSON.
- Computes base violations and scores for **all views** in the norm
  using wise.scoring.precompute.precompute_scores_for_all_views.
- Adds selected slice attributes at case level.
- Saves a single Parquet file that can be reused in the UI or notebooks.

Adjust paths and slice_cols as needed.
"""

from pathlib import Path
from typing import List

import pandas as pd

from wise.io.log_loader import load_event_log
from wise.io.norm_loader import load_norm_from_json
from wise.scoring.precompute import precompute_scores_for_all_views


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Paths
LOG_PATH = "data/BPI_Challenge_2019.csv"
NORM_PATH = "data/WISE_BPIC_Norm.json"
OUT_PATH = "data/BPI_Challenge_2019_precomputed.parquet"

# Core columns in the raw BPIC 2019 log
CASE_ID_COL = "case concept:name"
ACTIVITY_COL = "event concept:name"
TIMESTAMP_COL = "event time:timestamp"

# Slice columns you want to keep at case-level
# (adjust to match the actual column names in your CSV)
DESIRED_SLICE_COLS: List[str] = [
    "case Spend area text",
    "case Company",
    "case Document Type",
    "case Purch. Doc. Category",
    "case Item Type",
    "case Item Category",
    "case Spend classification text",
    "case Source",
    "case GR-Based Inv. Verif.",
    "case Goods Receipt",
    # add more as needed (e.g. derived duration bucket, day-of-week, etc.)
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading log from {LOG_PATH} ...")
    # Read with a permissive encoding; adjust if needed
    df_raw = pd.read_csv(LOG_PATH, encoding="latin1", low_memory=False)

    # Normalise timestamps etc. using WISE's loader
    df_log = load_event_log(
        df_raw,
        case_id_col=CASE_ID_COL,
        activity_col=ACTIVITY_COL,
        timestamp_col=TIMESTAMP_COL,
    )

    print(f"Log loaded: {len(df_log)} events, {df_log[CASE_ID_COL].nunique()} cases")
    print("Loading norm...")
    with open(NORM_PATH, "r", encoding="utf-8") as f:
        norm = load_norm_from_json(f)

    view_names = norm.get_view_names()
    print(f"Norm loaded with views: {view_names}")

    print("Computing base violations and scores for all views...")
    base_scores = precompute_scores_for_all_views(
        df=df_log,
        norm=norm,
        case_id_col=CASE_ID_COL,
        activity_col=ACTIVITY_COL,
        timestamp_col=TIMESTAMP_COL,
    )
    print(f"Base scores shape: {base_scores.shape}")

    # Determine which slice columns are actually present in the log
    available_slice_cols = [c for c in DESIRED_SLICE_COLS if c in df_log.columns]
    missing_slice_cols = [c for c in DESIRED_SLICE_COLS if c not in df_log.columns]

    if missing_slice_cols:
        print("Warning: the following slice columns were not found in the log and will be skipped:")
        for c in missing_slice_cols:
            print(f"  - {c}")

    if not available_slice_cols:
        print("No slice columns available; precomputed file will only contain scores.")
        slice_df = df_log[[CASE_ID_COL]].drop_duplicates()
    else:
        print("Using slice columns:")
        for c in available_slice_cols:
            print(f"  - {c}")
        slice_df = df_log[[CASE_ID_COL] + available_slice_cols].drop_duplicates()

    print("Adding slice attributes to base scores...")
    precomputed = base_scores.merge(slice_df, on=CASE_ID_COL, how="left")
    print(f"Precomputed table shape: {precomputed.shape}")

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving precomputed scores to {out_path} ...")
    precomputed.to_parquet(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
