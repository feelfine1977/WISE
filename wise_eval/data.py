"""
wise_eval.data — "Before computation: governance framing" (Fig. 2, top band).

Responsibilities (the two pre-computation steps of the WISE workflow):

  * Prepare event-log context — load the BPIC'19 CSV (or a synthetic
    stand-in), tidy the headers, parse timestamps, and map each item's
    `case Item Category` to one of the four flow types
    DF1 / DF2 / 2-way / Consignment.
  * Fix the case notion —
        case  sigma        = PO item  (`case concept:name` = PODoc_Item)
        governance unit    = PO header (`case Purchasing Document`)

Flow typing is a *prerequisite* for any normative assessment (paper
Sec. V-A): the same event pattern may be compliant in one flow type and
deviant in another, so the applicability map C_app is keyed on it.

The real BPIC'19 export already uses the canonical "case X" / "event X"
column names this pipeline expects — the only header quirk is a trailing
space on "eventID ", which _clean_columns() strips. Loading is LOUD:
load_raw_log() prints the file it loaded, or every path it checked before
falling back to the synthetic log.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG, RNG_SEED

# BPIC'19 "Item Category" -> flow type (paper Sec. V-A)
FLOW_MAP = {
    "3-way match, invoice after gr": "DF1",
    "3-way match, invoice before gr": "DF2",
    "2-way match": "2-way",
    "consignment": "Consignment",
}

# Columns the pipeline relies on by name (all present in the real export).
# `eventID` is handled separately — it is only a sort tiebreak and is
# synthesised if absent.
REQUIRED_COLUMNS = [
    "case concept:name", "case Purchasing Document", "case Company",
    "case Spend area text", "case Vendor", "case Item Type",
    "case Item Category", "event concept:name", "event org:resource",
    "event Cumulative net worth (EUR)", "event time:timestamp",
]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse internal whitespace and strip each header.

    Fixes the real file's `"eventID "` (trailing space) and any stray
    double-spaces, so the canonical names line up exactly.
    """
    out = df.copy()
    out.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in out.columns]
    return out


def _parse_timestamps(ser: pd.Series) -> pd.Series:
    """Parse BPIC'19 timestamps robustly.

    The real export is `dd-mm-yyyy HH:MM:SS.fff`, but not every row carries
    the `.fff` milliseconds. Inferring a single format would coerce the odd
    ones out to NaT, so we apply the fast explicit format to the bulk and
    fall back to per-element parsing only for the stragglers.

    Note: BPIC'19 uses a sentinel year (1948) as a missing-timestamp
    placeholder on some vendor-side events; these are parsed faithfully —
    surfacing such data-quality artefacts is the governance loop's job, not
    something to silently drop here.
    """
    out = pd.to_datetime(ser, errors="coerce", dayfirst=True,
                         format="%d-%m-%Y %H:%M:%S.%f")
    miss = out.isna() & ser.notna()
    if miss.any():
        out.loc[miss] = pd.to_datetime(ser[miss], errors="coerce",
                                       dayfirst=True, format="mixed")
    return out


def _is_human_resource(x) -> bool:
    """A resource is 'human' unless it is empty/NONE or a batch/system user."""
    s = str(x).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return False
    return not s.startswith("batch")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _resolve_csv_path():
    """First existing path in csv_candidates, else first *.csv inside any
    csv_search_dirs (so a non-standard filename still loads). Returns
    (path_or_None, checked_messages)."""
    checked = []
    for p in CONFIG["csv_candidates"]:
        if Path(p).exists():
            checked.append(f"FOUND    {p}")
            return Path(p), checked
        checked.append(f"missing  {p}")
    for folder in CONFIG.get("csv_search_dirs", []):
        d = Path(folder)
        if d.is_dir():
            hits = sorted(d.glob("*.csv"))
            if hits:
                checked.append(f"FOUND    {hits[0]}  (auto-detected in {d})")
                return hits[0], checked
            checked.append(f"missing  no *.csv in {d}")
        else:
            checked.append(f"missing  folder absent {d}")
    return None, checked


def _read_csv(path: Path) -> pd.DataFrame:
    """Read the CSV, retrying with ';' if it turns out to be semicolon-delimited."""
    df = pd.read_csv(path, encoding=CONFIG["encoding"], low_memory=False)
    if df.shape[1] == 1:
        df = pd.read_csv(path, encoding=CONFIG["encoding"], sep=";", low_memory=False)
    return df


def load_raw_log(verbose: bool = True):
    """Load the real BPIC'19 CSV if available; otherwise synthesise.

    Returns (df_raw, source, path) where source is "BPIC19" or "SYNTHETIC".
    Set CONFIG['require_real_data']=True to raise instead of synthesising.
    """
    csv_path, checked = _resolve_csv_path()
    if csv_path is not None:
        if verbose:
            print(f"[data] loading REAL BPIC'19 log: {csv_path}")
        df_raw = _read_csv(csv_path)
        if verbose:
            print(f"[data] read {len(df_raw):,} rows x {df_raw.shape[1]} columns")
        return df_raw, "BPIC19", csv_path

    msg = "[data] real BPIC'19 CSV NOT found. Checked:\n  " + "\n  ".join(checked)
    if CONFIG.get("require_real_data", False):
        raise FileNotFoundError(
            msg + "\n[data] require_real_data=True -> aborting instead of "
            "synthesising. Put the CSV in "
            f"{CONFIG.get('csv_search_dirs', ['the data folder'])[0]}.")
    if verbose:
        print(msg)
        print("[data] -> falling back to SYNTHETIC log with planted hotspots.")
    return generate_synthetic_bpic19(CONFIG["synthetic_n_docs"]), "SYNTHETIC", None


# --------------------------------------------------------------------------
# Preparation (fixes the case notion, derives the governance keys)
# --------------------------------------------------------------------------
def prepare_log(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean, type, and index the event log.

    Adds: case_id (PO item), purchasing_document (governance unit),
    event_activity, flow_type (DF1/DF2/2-way/Consignment), is_human_resource,
    case_start_quarter (temporal drill-down key for the governance loop).
    """
    df = _clean_columns(df_raw)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "BPIC'19 log is missing required columns: " + str(missing) +
            "\nColumns present: " + str(list(df.columns)[:40]) +
            "\nIf your export uses different headers, rename them to the "
            "canonical 'case X' / 'event X' names or tell me the headers.")

    # eventID is only a stable sort tiebreak — synthesise if the file lacks it.
    if "eventID" not in df.columns:
        df["eventID"] = np.arange(len(df))

    df["event time:timestamp"] = _parse_timestamps(df["event time:timestamp"])
    n_nat = int(df["event time:timestamp"].isna().sum())
    if n_nat:
        print(f"[data] note: {n_nat:,} timestamps could not be parsed (set to NaT)")

    # Slice/drill-down keys: a missing value (e.g. empty spend area) is a
    # real, ownership-relevant category — make it explicit instead of NaN so
    # grouping, masks, labels, and figures treat it as a normal slice.
    for col in ["case Company", "case Spend area text", "case Vendor", "case Item Type"]:
        if col in df.columns:
            df[col] = df[col].fillna("(missing)").replace({"": "(missing)", "nan": "(missing)"})

    df["case_id"] = df["case concept:name"].astype(str).str.strip()
    df["purchasing_document"] = df["case Purchasing Document"].astype(str).str.strip()
    df["event_activity"] = df["event concept:name"].astype(str).str.strip()
    df["flow_type_raw"] = df["case Item Category"].astype(str).str.strip()
    df["flow_type"] = df["flow_type_raw"].str.lower().map(FLOW_MAP).fillna(df["flow_type_raw"])
    df["is_human_resource"] = df["event org:resource"].map(_is_human_resource)

    df = df.sort_values(
        ["case Item Category", "case_id", "event time:timestamp", "eventID"],
        kind="mergesort").reset_index(drop=True)

    case_start = df.groupby("case_id")["event time:timestamp"].min().rename("case_start_ts")
    case_quarter = case_start.dt.to_period("Q").astype(str).rename("case_start_quarter")
    df = df.merge(case_quarter.reset_index(), on="case_id", how="left")
    return df


# --------------------------------------------------------------------------
# Synthetic fallback — testability by design
# --------------------------------------------------------------------------
def generate_synthetic_bpic19(n_docs: int = 1500, seed: int = RNG_SEED) -> pd.DataFrame:
    """Generate a BPIC'19-schema P2P log with three *planted* hotspots.

    The planted patterns reproduce the typology the paper finds on the real
    log, so the pipeline doubles as its own validation: it must rediscover
    and correctly type them.

      (1) RESERVOIR  — companyID_0000 x Packaging: moderate per-case
          closure/ageing deviation at very large volume (L1/L3).
      (2) MECHANISM  — companyID_0000 x Logistics: goods-receipt
          fragmentation (repeated GR postings, L4); a share share identical
          timestamps to emulate the extraction-level event replication
          BPIC'19 shows for service items — the governance loop must flag it.
      (3) SEVERITY   — companyID_0003 x Real Estate: small slice with
          intense approval-change churn (L4).
    """
    rng = np.random.default_rng(seed)
    companies = ["companyID_0000", "companyID_0001", "companyID_0002", "companyID_0003"]
    spend_areas = ["Packaging", "Logistics", "Real Estate", "Additives",
                   "Energy", "IT", "Marketing", "Sales"]
    item_types = ["Standard", "Service", "Consignment", "Limit", "Subcontracting"]
    doc_types = ["Standard PO", "Framework order", "EC Purchase order"]
    flows_raw = ["3-way match, invoice before GR", "3-way match, invoice after GR",
                 "2-way match", "Consignment"]
    flow_p = [0.80, 0.10, 0.02, 0.08]   # BPIC'19 is dominated by DF2
    vendors = [f"vendorID_{i:04d}" for i in range(60)]
    users = [f"user_{i:03d}" for i in range(40)] + [f"batch_{i:02d}" for i in range(6)]

    rows, eid = [], 0
    t0 = pd.Timestamp("2018-01-01")
    extraction_cut = t0 + pd.Timedelta(days=455)   # truncation -> right-censoring

    for d in range(n_docs):
        company = rng.choice(companies, p=[0.62, 0.16, 0.12, 0.10])
        if company == "companyID_0000":
            spend = rng.choice(spend_areas, p=[0.40, 0.22, 0.04, 0.10, 0.06, 0.06, 0.06, 0.06])
        elif company == "companyID_0003":
            spend = rng.choice(spend_areas, p=[0.10, 0.08, 0.38, 0.10, 0.10, 0.08, 0.08, 0.08])
        else:
            spend = rng.choice(spend_areas)
        vendor = rng.choice(vendors)
        doc_type = rng.choice(doc_types, p=[0.85, 0.10, 0.05])
        item_type = rng.choice(item_types, p=[0.6, 0.2, 0.08, 0.06, 0.06])
        doc_id = f"docID_{d:06d}"
        n_items = int(rng.integers(1, 5))

        for it in range(1, n_items + 1):
            flow_raw = rng.choice(flows_raw, p=flow_p)
            case_id = f"{doc_id}_{it:03d}"
            start = t0 + pd.Timedelta(days=float(rng.uniform(0, 445)))
            net = float(rng.lognormal(7.5, 1.6))
            hotspot_pack = company == "companyID_0000" and spend == "Packaging"
            hotspot_log = company == "companyID_0000" and spend == "Logistics"
            hotspot_re = company == "companyID_0003" and spend == "Real Estate"

            evs = [("Create Purchase Order Item", 0.0)]
            if rng.random() < 0.3 or hotspot_re:                       # approval churn
                n_appr = int(rng.integers(2, 5)) if (hotspot_re and rng.random() < 0.85) else 1
                approver = rng.choice(users[:40])
                for k in range(n_appr):
                    evs.append(("Change Approval for Purchase Order", 0.2 + 0.5 * k, approver))
            if "3-way" in flow_raw or flow_raw == "Consignment":      # goods receipts
                n_gr = 1
                if hotspot_log and rng.random() < 0.65:
                    n_gr = int(rng.integers(3, 8))
                elif rng.random() < 0.12:
                    n_gr = int(rng.integers(2, 4))
                replicate = hotspot_log and n_gr >= 3 and rng.random() < 0.5
                if replicate:
                    n_gr = max(n_gr, int(rng.integers(5, 9)))
                copies = int(rng.integers(3, 5)) if replicate else 1
                gr_clerk = (rng.choice(users[40:]) if hotspot_log else rng.choice(users[:40])) \
                    if n_gr >= 3 else None
                for k in range(n_gr):
                    off = 2 + 3 * k + (0.0 if replicate else rng.uniform(0, 2))
                    fixed_res = gr_clerk if gr_clerk is not None else (
                        rng.choice(users) if replicate else None)
                    for _copy in range(copies):
                        evs.append(("Record Goods Receipt", float(off), fixed_res))
            if flow_raw != "Consignment":                             # invoicing
                inv_lag = rng.uniform(1, 12)
                if flow_raw == "3-way match, invoice before GR":
                    inv_lag = rng.uniform(0.5, 4)
                evs.append(("Record Invoice Receipt", 2 + inv_lag))
                if flow_raw == "3-way match, invoice before GR" and rng.random() < 0.5:
                    evs.append(("Remove Payment Block", 4 + inv_lag + rng.uniform(0, 6)))
                missing_clear_p = 0.45 if hotspot_pack else 0.10
                if rng.random() > missing_clear_p:
                    clear_lag = rng.uniform(45, 140) if hotspot_pack else rng.uniform(12, 60)
                    evs.append(("Clear Invoice", 4 + inv_lag + clear_lag))
                if rng.random() < 0.05:
                    evs.append(("Cancel Invoice Receipt", 3 + inv_lag))
                if rng.random() < 0.04:
                    evs.append(("Vendor creates debit memo", 6 + inv_lag))
            if rng.random() < 0.08:
                evs.append(("Change Price", 1.0))
            if rng.random() < 0.08:
                evs.append(("Change Quantity", 1.2))
            if rng.random() < 0.03:
                evs.append(("Delete Purchase Order Item", 1.5))

            for ev in evs:
                act, off = ev[0], ev[1]
                fixed_res = ev[2] if len(ev) > 2 else None
                res = fixed_res if fixed_res is not None else (
                    rng.choice(users[:40]) if rng.random() < 0.55 else rng.choice(users[40:]))
                rows.append({
                    "eventID": f"ev_{eid:08d}",
                    "case Spend area text": spend,
                    "case Sub spend area text": f"{spend} / sub{int(rng.integers(1, 4))}",
                    "case Company": company,
                    "case Purchasing Document": doc_id,
                    "case Vendor": vendor,
                    "case Item Type": item_type,
                    "case Item Category": flow_raw,
                    "case Item": f"{it:03d}",
                    "case Document Type": doc_type,
                    "case Purch. Doc. Category name": "Purchase order",
                    "case Spend classification text": rng.choice(["NPR", "PR"]),
                    "case Source": "SAP",
                    "case Name": f"name_{company}",
                    "case GR-Based Inv. Verif.": flow_raw == "3-way match, invoice after GR",
                    "case Goods Receipt": "3-way" in flow_raw or flow_raw == "Consignment",
                    "case concept:name": case_id,
                    "event org:resource": res,
                    "event concept:name": act,
                    "event Cumulative net worth (EUR)": net * (1 + rng.normal(0, 0.05)),
                    "event time:timestamp": start + pd.Timedelta(days=float(off)),
                })
                eid += 1
    out = pd.DataFrame(rows)
    out = out[out["event time:timestamp"] <= extraction_cut].reset_index(drop=True)
    out["event time:timestamp"] = out["event time:timestamp"].dt.strftime("%d-%m-%Y %H:%M:%S")
    return out
