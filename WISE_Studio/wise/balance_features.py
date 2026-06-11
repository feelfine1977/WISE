"""Optional per-activity numeric totals required by `balance` constraints.

Called from the pipeline only when the norm contains balance constraints, so
ordinary runs stay fast. Adds columns total__<attr>__<activity_slug>.
"""
import pandas as pd
from .mapping import slug


def add_balance_totals(fc: pd.DataFrame, events: pd.DataFrame, norm: dict) -> pd.DataFrame:
    pairs = set()
    for c in norm.get("constraints", []):
        if c.get("type") == "balance":
            for a in list(c.get("activities_x", [])) + list(c.get("activities_y", [])):
                pairs.add((c["attribute"], a))
    if not pairs:
        return fc
    fc = fc.copy()
    for attr, act in pairs:
        if attr not in events.columns:
            continue
        sub = events[events["activity"] == act]
        tot = sub.groupby("case_id")[attr].sum()
        fc[f"total__{slug(attr)}__{slug(act)}"] = fc["case_id"].map(tot).fillna(0.0)
    return fc
