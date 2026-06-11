"""
wise_eval.config — every methodological choice in one auditable place.

WISE's governance requirement (paper Sec. II-E gap 3, Sec. III) is that
decision-layer artefacts — norms, weights, aggregation choices — are
explicit, versionable, and auditable. Hard-coding "magic numbers" deep
inside an analysis is exactly the anti-pattern WISE is designed against,
so all knobs live here.
"""
from pathlib import Path

RNG_SEED = 42

# Repository root = the folder that contains this package (wise_eval/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ===========================================================================
#  >>> THE ONLY THING YOU NORMALLY NEED TO CONFIGURE <<<
#
#  DATA_DIR = the folder that contains your BPIC'19 CSV.
#  Default: the  data/  folder next to the notebooks. Any *.csv inside it
#  (or one level of subfolders) is found automatically — the filename does
#  not have to be exactly "BPI_Challenge_2019.csv".
#
#  You can also override this per session from a notebook without editing
#  this file:   CONFIG["csv_search_dirs"].insert(0, r"C:\my\data\folder")
# ===========================================================================
DATA_DIR = PROJECT_ROOT / "data"

CONFIG = {
    # ---------------------------------------------------------------- data
    # Real BPIC'19 log: 1,595,923 events / 251,734 PO items /
    # 76,349 purchasing documents. Exact-path candidates are tried first,
    # then every folder in csv_search_dirs is scanned for *.csv.
    "csv_candidates": [
        str(DATA_DIR / "BPI_Challenge_2019.csv"),
        str(DATA_DIR / "data_BPIC_2019" / "BPI_Challenge_2019.csv"),
    ],
    "csv_search_dirs": [str(DATA_DIR), str(DATA_DIR / "data_BPIC_2019")],
    # Set True to raise a clear error instead of silently synthesising when
    # the real CSV cannot be found.
    "require_real_data": False,
    "encoding": "latin-1",
    "synthetic_n_docs": 1500,

    # ---------------------------------------------------------------- norm
    "norm_path": str(PROJECT_ROOT / "norm" / "bpic19_norm_v1.json"),

    # ----------------------------------------------------------- slicing g
    # Primary slicing function g(σ) = company × spend area (paper Sec. V-A):
    # these keys map to governance structures with accountable owners.
    "slice_keys": ["case Company", "case Spend area text"],
    # Drill-down keys for validation and follow-up (vendor Pareto, quarter
    # trend). They are NOT part of the primary ranking — adding keys
    # multiplies slices and erodes statistical support per slice.
    "drilldown_keys": ["case Vendor", "case Item Type", "case_start_quarter"],

    # ------------------------------------------------- prioritisation (PI)
    # PI^(p)_s = n_s · ( mu_bar^(p) - mu_s^(p) )_+        (paper Sec. IV-E)
    # Shrinkage constant gamma: small slices are pulled toward the global
    # mean so noisy gaps cannot dominate the backlog. gamma = 20 means a
    # 20-case slice keeps only half of its observed gap.
    "shrinkage_gamma": 20.0,

    # ------------------------------------------------------ governance loop
    # Right-censoring window (paper Table X notes): an invoice-bearing item
    # without Clear Invoice that is still "active" within the final 60 days
    # of the observation window may simply lack time to finish.
    "censoring_window_days": 60,
    # Event-replication flag (paper Table X notes): item-level ratio of
    # events per *distinct timestamp* above 2 indicates extraction-level
    # duplication rather than genuine repeated work.
    "replication_ratio_flag": 2.0,

    # ----------------------------------------------------------- reporting
    "top_k_documents": 20,     # shortlist size for Jaccard view comparison
    "pareto_threshold": 0.80,  # "k of K carry 80% of penalty mass"
    "figures_dir": str(PROJECT_ROOT / "figures" / "generated"),
}
