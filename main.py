import pandas as pd

from wise.io.log_loader import load_event_log
from wise.io.norm_loader import load_norm_from_json
from wise.scoring.scoring import compute_case_scores
from wise.scoring.slices import aggregate_slices


# --------------------------------------------------------------------
# Configuration – adjust here for other logs
# --------------------------------------------------------------------

# Read all from settings.json if needed
settings_path = "settings.json"
settings = pd.read_json(settings_path, typ="series")
# Paths
LOG_PATH = settings["log_path"]  # e.g. "data/BPI_2019.csv"
NORM_PATH = settings["norm_path"]  # e.g. "data/WISE_BPIC_Norm.json"

# Raw column names in BPIC_2019.csv
CASE_ID_COL = settings["case_id_col"]  # e.g. "case:concept:name"
ACTIVITY_COL = settings["activity_col"]  # e.g. "event:concept:name"
TIMESTAMP_COL = settings["timestamp_col"]  # e.g. "time:timestamp"

# Slice attributes for BPIC_2019 – change as needed
SLICE_COLS = settings["SLICE_COLS"]  # e.g. ["case:Company", "case:Document Type"]

# View name to use from the norm (e.g. "Finance" or "Logistics")
DEFAULT_VIEW = settings["default_view"]  # e.g. "Finance"

# Empirical-Bayes shrinkage parameter; set to 0 to disable
SHRINK_K = 50.0


def main():
    # 1. Load event log
    print(f"Loading event log from {LOG_PATH} ...")
    df = load_event_log(
        LOG_PATH,
        case_id_col=CASE_ID_COL,
        activity_col=ACTIVITY_COL,
        timestamp_col=TIMESTAMP_COL,
    )
    print(f"Loaded {len(df)} events for {df[CASE_ID_COL].nunique()} cases.")

    # 2. Load norm
    print(f"Loading norm from {NORM_PATH} ...")
    with open(NORM_PATH, "r", encoding="utf-8") as f:
        norm = load_norm_from_json(f)

    view_names = norm.get_view_names()
    if not view_names:
        raise ValueError("Norm contains no views.")
    if DEFAULT_VIEW not in view_names:
        print(f"View '{DEFAULT_VIEW}' not found in norm, using first view: {view_names[0]}")
        view_name = view_names[0]
    else:
        view_name = DEFAULT_VIEW

    print(f"Using view: {view_name}")

    # 3. Case-level scores
    print("Computing case-level scores ...")
    case_scores = compute_case_scores(
        df=df,
        norm=norm,
        view_name=view_name,
        case_id_col=CASE_ID_COL,
        activity_col=ACTIVITY_COL,
        timestamp_col=TIMESTAMP_COL,
    )
    print("Case scores head:")
    print(case_scores.head())

    # 4. Slice-level aggregation and PI
    if SLICE_COLS:
        print(f"Aggregating scores by slices: {SLICE_COLS}")
        slice_summary = aggregate_slices(
            df_scores=case_scores,
            df_log=df,
            case_id_col=CASE_ID_COL,
            slice_cols=SLICE_COLS,
            shrink_k=SHRINK_K,
        )
        print("Top slices by Priority Index:")
        print(slice_summary.head(20))

        # Optional: save to CSV for further analysis
        slice_summary.to_csv("data/WISE_slice_summary.csv", index=False)
        case_scores.to_csv("data/WISE_case_scores.csv", index=False)
        print("Saved WISE_slice_summary.csv and WISE_case_scores.csv under data/.")
    else:
        print("No slice columns configured; slice-level PI not computed.")


if __name__ == "__main__":
    main()
