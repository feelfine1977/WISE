"""Five generic constraint types with threshold–saturation, evaluated on case
features. Applicability is a generic attribute filter, so no process-specific
logic lives in the core.

Constraint schema (norm.json, per entry in "constraints"):
  common: id, layer, type, description?, within_layer_weight? (default 1),
          applicability: [ {"column": <case attr>, "values": [..]} , ... ]  (AND)
  presence:    activity, min_count (default 1)
  lag:         from_activity, to_activity, threshold_days, saturation_days
  singularity: activity, allowed, saturation
  exclusion:   activity
  balance:     attribute, activities_x, activities_y, tolerance, saturation
"""
import numpy as np
import pandas as pd

from .mapping import slug


def sat(z, threshold, width):
    """Threshold–saturation: 0 below threshold, linear, capped at 1."""
    z = np.asarray(z, dtype=float)
    return np.clip((z - threshold) / max(width, 1e-9), 0.0, 1.0)


def _cnt(fc, activity):
    col = f"count__{slug(activity)}"
    return fc[col].fillna(0) if col in fc.columns else pd.Series(0.0, index=fc.index)


def _first(fc, activity):
    col = f"first_ts__{slug(activity)}"
    return fc[col] if col in fc.columns else pd.Series(pd.NaT, index=fc.index)


def applicability_mask(fc: pd.DataFrame, constraint: dict) -> pd.Series:
    m = pd.Series(True, index=fc.index)
    for f in constraint.get("applicability", []) or []:
        col, vals = f.get("column"), f.get("values", [])
        if col in fc.columns and vals:
            m &= fc[col].astype(str).isin([str(v) for v in vals])
    return m


def evaluate_constraint(fc: pd.DataFrame, c: dict) -> pd.Series:
    """Bounded violation ν_c(σ) ∈ [0,1]; NaN where the constraint is not applicable."""
    t = c["type"]
    if t == "presence":
        m = max(int(c.get("min_count", 1)), 1)
        v = 1 - np.minimum(_cnt(fc, c["activity"]) / m, 1.0)
    elif t == "exclusion":
        v = (_cnt(fc, c["activity"]) > 0).astype(float)
    elif t == "singularity":
        v = sat(_cnt(fc, c["activity"]), float(c["allowed"]), float(c["saturation"]))
        v = pd.Series(v, index=fc.index)
    elif t == "lag":
        ta, tb = _first(fc, c["from_activity"]), _first(fc, c["to_activity"])
        lag_days = (tb - ta).dt.total_seconds() / 86400
        v = pd.Series(sat(lag_days.fillna(np.inf).clip(lower=0),
                          float(c["threshold_days"]), float(c["saturation_days"])),
                      index=fc.index)
        v[ta.isna() | tb.isna() | (lag_days < 0)] = 1.0  # missing/ordering breach = max
    elif t == "balance":
        a = slug(c["attribute"])
        # per-case totals over the two activity sets need event data; approximated
        # at case level via total__ columns when both sets cover all events,
        # otherwise computed from per-activity share is not possible — so balance
        # uses the case-level totals of the attribute split by activity sets,
        # prepared in features as total__<attr>__<actslug> when configured.
        x = sum((fc.get(f"total__{a}__{slug(ax)}", pd.Series(0.0, index=fc.index)).fillna(0)
                 for ax in c["activities_x"]), start=pd.Series(0.0, index=fc.index))
        y = sum((fc.get(f"total__{a}__{slug(ay)}", pd.Series(0.0, index=fc.index)).fillna(0)
                 for ay in c["activities_y"]), start=pd.Series(0.0, index=fc.index))
        d = (x - y).abs() / np.maximum(np.maximum(x, y), 1e-9)
        v = pd.Series(sat(d, float(c["tolerance"]), float(c["saturation"])), index=fc.index)
    else:
        raise ValueError(f"unknown constraint type: {t}")
    v = pd.Series(np.asarray(v, dtype=float), index=fc.index)
    v[~applicability_mask(fc, c)] = np.nan
    return v


def build_violation_matrix(fc: pd.DataFrame, norm: dict) -> pd.DataFrame:
    """case_id + one ν column per constraint id (NaN = not applicable)."""
    out = pd.DataFrame({"case_id": fc["case_id"]})
    for c in norm["constraints"]:
        out[c["id"]] = evaluate_constraint(fc, c).values
    return out
