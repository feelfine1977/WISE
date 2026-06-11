"""Load any CSV and canonicalise it via a mapping."""
import warnings
from pathlib import Path

import pandas as pd

CANON = ["case_id", "activity", "timestamp", "resource"]


def read_csv_any(path_or_buffer, encoding="utf-8", nrows=None, usecols=None):
    """Tolerant CSV read: tries the declared encoding then common fallbacks.

    `nrows` enables cheap sample reads for profiling; `usecols` prunes the
    parse to only the mapped columns — the main memory lever for large files.
    """
    encodings = [encoding] + [e for e in ("utf-8", "latin-1", "cp1252") if e != encoding]
    last = None
    for enc in encodings:
        try:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            return pd.read_csv(path_or_buffer, encoding=enc, nrows=nrows,
                               usecols=usecols, low_memory=False), enc
        except (UnicodeDecodeError, UnicodeError) as e:
            last = e
        except ValueError as e:           # usecols mismatch → clearer error
            raise ValueError(f"Column selection failed ({e}). "
                             "Does the mapping match this file?") from e
    raise last


def mapped_columns(mapping: dict) -> list:
    """The only columns the analysis needs — used to prune large reads."""
    cols = [mapping.get(r) for r in ("case_id", "activity", "timestamp",
                                     "resource", "document_id")]
    cols += list(mapping.get("dimensions", []))
    cols += list(mapping.get("numeric_attributes", []))
    return [c for c in dict.fromkeys(cols) if c]   # dedupe, drop None


def load_sample(path_or_buffer, encoding="utf-8", sample_rows=150_000):
    """Fast sample read for profiling/auto-detection of big files."""
    return read_csv_any(path_or_buffer, encoding=encoding, nrows=sample_rows)


def load_mapped(path_or_buffer, mapping: dict):
    """Memory-lean full load: parse ONLY the mapped columns, then canonicalise
    (which also downcasts repetitive text columns to category)."""
    df, enc = read_csv_any(path_or_buffer, encoding=mapping.get("encoding", "utf-8"),
                           usecols=mapped_columns(mapping))
    return canonicalize(df, mapping), enc


def canonicalize(df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Return event frame with canonical columns + kept dimensions/attributes.

    Canonical: case_id (str), activity (str), timestamp (datetime, sorted),
    resource (str or '(none)'). Dimensions keep their original column names;
    missing dimension values become '(missing)' so they group/label cleanly.
    """
    df = df_raw.copy()
    df["case_id"] = df[mapping["case_id"]].astype(str).str.strip()
    df["activity"] = df[mapping["activity"]].astype(str).str.strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = pd.to_datetime(df[mapping["timestamp"]], errors="coerce",
                            format=mapping.get("timestamp_format") or "mixed")
        if ts.isna().mean() > 0.5:  # retry dayfirst
            ts = pd.to_datetime(df[mapping["timestamp"]], errors="coerce",
                                format="mixed", dayfirst=True)
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"])
    res_col = mapping.get("resource")
    df["resource"] = (df[res_col].astype(str).fillna("(none)")
                      if res_col and res_col in df.columns else "(none)")
    for dcol in list(mapping.get("dimensions", [])):
        if dcol in df.columns:
            df[dcol] = df[dcol].fillna("(missing)").replace({"": "(missing)", "nan": "(missing)"}).astype(str)
    doc = mapping.get("document_id")
    if doc and doc in df.columns:
        df["document_id"] = df[doc].astype(str)
    for a in mapping.get("numeric_attributes", []):
        if a in df.columns:
            df[a] = pd.to_numeric(df[a], errors="coerce")
    keep = (CANON + (["document_id"] if "document_id" in df.columns else [])
            + [c for c in mapping.get("dimensions", []) if c in df.columns]
            + [a for a in mapping.get("numeric_attributes", []) if a in df.columns])
    out = df[keep].sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    # Categorical dtypes: repetitive text columns shrink 5–20× on large logs.
    for c in (["activity", "resource", "document_id"]
              + [d for d in mapping.get("dimensions", []) if d in out.columns]):
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out
