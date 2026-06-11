"""Log mapping: which columns mean what. Schema + auto-detection + validation.

A mapping is a plain dict (saved as log_mapping.json):
{
  "case_id":   "<column>",          # required
  "activity":  "<column>",          # required
  "timestamp": "<column>",          # required
  "resource":  "<column or null>",  # optional
  "dimensions": ["<col>", ...],     # business slicing attributes (company, region, ...)
  "numeric_attributes": ["<col>", ...],   # event-level numeric values (amounts, qty)
  "document_id": "<column or null>",      # optional governance roll-up (e.g. PO header)
  "encoding": "utf-8",
  "timestamp_format": null          # null = pandas auto-parse
}
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("case_id", "activity", "timestamp")


def empty_mapping():
    return {"case_id": None, "activity": None, "timestamp": None,
            "resource": None, "dimensions": [], "numeric_attributes": [],
            "document_id": None, "encoding": "utf-8", "timestamp_format": None}


def load_mapping(path):
    m = empty_mapping(); m.update(json.load(open(path)))
    return m


def save_mapping(mapping, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(mapping, open(path, "w"), indent=2)


# ---------------------------------------------------------------- detection
def _ts_parse_rate(s: pd.Series) -> float:
    """Share of a sample that parses as a datetime."""
    sample = s.dropna().astype(str).head(300)
    if len(sample) == 0:
        return 0.0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=False)
        if parsed.notna().mean() < 0.5:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=True)
    return float(parsed.notna().mean())


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column profile used by detection and shown to the user."""
    n = max(len(df), 1)
    rows = []
    for c in df.columns:
        s = df[c]
        nun = int(s.nunique(dropna=True))
        num_rate = float(pd.to_numeric(s, errors="coerce").notna().mean())
        rows.append({"column": c, "dtype": str(s.dtype), "n_unique": nun,
                     "unique_ratio": nun / n, "null_share": float(s.isna().mean()),
                     "numeric_rate": num_rate, "ts_parse_rate": _ts_parse_rate(s),
                     "example": str(s.dropna().iloc[0])[:48] if s.notna().any() else ""})
    return pd.DataFrame(rows)


def detect_mapping(df: pd.DataFrame):
    """Best-guess mapping + per-role confidence in [0,1]. User confirms/overrides.

    Detection order matters: timestamp (parse rate) → case id (cardinality)
    → activity (VARIES within a case, low global cardinality)
    → dimensions (CONSTANT within a case). Within-case variability is the
    structural signal that separates an activity from a case attribute.
    """
    prof = profile_columns(df)
    m, conf = empty_mapping(), {}

    def name_hit(c, words):
        lc = str(c).lower()
        return any(w in lc for w in words)

    # --- timestamp: best datetime parse rate (+ name hint) -----------------
    p = prof.assign(sc=prof["ts_parse_rate"]
                    + prof["column"].map(lambda c: 0.15 if name_hit(c, ("time", "date", "stamp", "when")) else 0))
    p = p.sort_values("sc", ascending=False)
    if p.iloc[0]["ts_parse_rate"] > 0.5:
        m["timestamp"] = p.iloc[0]["column"]; conf["timestamp"] = float(min(p.iloc[0]["sc"], 1))

    # --- case id: high cardinality, id-like (+ name hint) ------------------
    cand = prof[(prof["column"] != m["timestamp"]) & (prof["n_unique"] > 1)].copy()
    cand["sc"] = cand["unique_ratio"] + cand["column"].map(
        lambda c: 0.35 if name_hit(c, ("case", "_id", " id", "id_", "concept:name",
                                       "ticket", "order", "number")) else 0)
    cand = cand.sort_values("sc", ascending=False)
    if len(cand):
        m["case_id"] = cand.iloc[0]["column"]; conf["case_id"] = float(min(cand.iloc[0]["sc"], 1))

    # --- within-case variability (sampled): activity varies, dims don't ----
    var_share = {}
    if m["case_id"]:
        sample_cases = df[m["case_id"]].dropna().unique()[:1500]
        sub = df[df[m["case_id"]].isin(sample_cases)]
        for c in df.columns:
            if c in (m["case_id"], m["timestamp"]):
                continue
            try:
                nun = sub.groupby(m["case_id"], sort=False)[c].nunique()
                var_share[c] = float((nun > 1).mean())   # share of cases where col varies
            except TypeError:
                var_share[c] = 0.0

    # --- activity: varies within case, modest global cardinality -----------
    cand = prof[(~prof["column"].isin([m["timestamp"], m["case_id"]]))
                & (prof["n_unique"].between(2, 1000)) & (prof["numeric_rate"] < 0.5)
                & (prof["ts_parse_rate"] < 0.5)].copy()
    if len(cand):
        cand["sc"] = (cand["column"].map(lambda c: var_share.get(c, 0.0)) * 1.0
                      + (1 - cand["unique_ratio"]) * 0.3
                      + cand["column"].map(lambda c: 0.3 if name_hit(
                          c, ("activity", "concept:name", "event", "task", "action",
                              "step", "status", "stage")) else 0))
        best = cand.sort_values("sc", ascending=False).iloc[0]
        m["activity"] = best["column"]; conf["activity"] = float(min(best["sc"], 1))

    used = {m["case_id"], m["activity"], m["timestamp"]}

    # --- resource: name hint, or varies-within-case string with many values -
    res_cands = [c for c in df.columns if c not in used and name_hit(
        c, ("resource", "user", "org:", "agent", "operator", "handler",
            "worker", "employee", "performer"))]
    if not res_cands:
        res_cands = [c for c, v in sorted(var_share.items(), key=lambda kv: -kv[1])
                     if c not in used and v > 0.3
                     and prof.set_index("column").loc[c, "numeric_rate"] < 0.5
                     and prof.set_index("column").loc[c, "ts_parse_rate"] < 0.5]
    if res_cands:
        m["resource"] = res_cands[0]; conf["resource"] = 0.7
        used.add(m["resource"])

    # --- dimensions: categorical AND constant within a case ----------------
    dims = prof[(~prof["column"].isin(used)) & (prof["n_unique"].between(2, 200))
                & (prof["numeric_rate"] < 0.7) & (prof["ts_parse_rate"] < 0.5)].copy()
    dims = dims[dims["column"].map(lambda c: var_share.get(c, 0.0) < 0.10)]
    m["dimensions"] = list(dims.sort_values("n_unique")["column"].head(8))

    # --- numeric attributes -------------------------------------------------
    nums = prof[(~prof["column"].isin(used)) & (prof["numeric_rate"] > 0.9)
                & (prof["n_unique"] > 20)]
    m["numeric_attributes"] = list(nums["column"].head(5))

    # --- optional document/roll-up id ---------------------------------------
    if m["case_id"]:
        case_card = prof.set_index("column").loc[m["case_id"], "n_unique"]
        doc = prof[(~prof["column"].isin(used))
                   & (prof["n_unique"].between(20, max(int(case_card * 0.9), 21)))
                   & prof["column"].map(lambda c: name_hit(
                       c, ("document", "purchas", "header", "parent")))]
        if len(doc):
            m["document_id"] = doc.iloc[0]["column"]
    return m, conf, prof


def validate_mapping(df: pd.DataFrame, mapping: dict):
    """Return list of (level, message); level in {'error','warn','ok'}."""
    msgs = []
    for r in REQUIRED:
        col = mapping.get(r)
        if not col:
            msgs.append(("error", f"Required role '{r}' is not set.")); continue
        if col not in df.columns:
            msgs.append(("error", f"'{r}' column '{col}' not found in the file.")); continue
    if any(l == "error" for l, _ in msgs):
        return msgs
    ts_rate = _ts_parse_rate(df[mapping["timestamp"]])
    msgs.append(("ok" if ts_rate > 0.95 else "warn",
                 f"Timestamp '{mapping['timestamp']}' parses at {ts_rate:.0%}."))
    n_act = df[mapping["activity"]].nunique()
    msgs.append(("ok" if 2 <= n_act <= 1000 else "warn",
                 f"Activity '{mapping['activity']}' has {n_act} distinct values."))
    n_case = df[mapping["case_id"]].nunique()
    msgs.append(("ok" if n_case > 1 else "error",
                 f"Case id '{mapping['case_id']}' has {n_case:,} distinct cases "
                 f"({len(df)/max(n_case,1):.1f} events/case)."))
    for d in mapping.get("dimensions", []):
        if d not in df.columns:
            msgs.append(("warn", f"Dimension '{d}' not found — it will be ignored."))
    for a in mapping.get("numeric_attributes", []):
        if a not in df.columns:
            msgs.append(("warn", f"Numeric attribute '{a}' not found — ignored."))
    return msgs


def slug(activity: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(activity).lower()).strip("_")
