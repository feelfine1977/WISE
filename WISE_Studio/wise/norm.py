"""Generic norm schema: layers, constraints, views — plus validation and a
log-driven starter-norm suggester that powers the guided norm builder.

norm.json:
{
  "name": "...", "version": "1.0",
  "layers": {"L1": {"name": "Completion & closure"}, ...},
  "constraints": [ ... see constraints.py ... ],
  "views": {"Default": {"layer_weights": {"L1": 0.3, ...}}}
}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .mapping import slug

DEFAULT_LAYERS = {
    "L1": {"name": "Completion & closure"},
    "L2": {"name": "Timeliness & ageing"},
    "L3": {"name": "Rework & repetition"},
    "L4": {"name": "Exceptions & corrections"},
    "L5": {"name": "Effort & complexity"},
}

EXCEPTION_WORDS = ("cancel", "delete", "reject", "remove", "reverse",
                   "return", "decline", "withdraw", "change", "block")


def load_norm(path):
    return json.load(open(path))


def save_norm(norm, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(norm, open(path, "w"), indent=2)


def validate_norm(norm: dict, fc: pd.DataFrame):
    """(level, message) checks: structure, referenced activities, weights."""
    msgs = []
    if not norm.get("constraints"):
        return [("error", "The norm has no constraints yet.")]
    layer_ids = set(norm.get("layers", {}))
    have_acts = {c[len("count__"):] for c in fc.columns if c.startswith("count__")}
    ids = set()
    for c in norm["constraints"]:
        cid = c.get("id", "?")
        if cid in ids:
            msgs.append(("error", f"Duplicate constraint id '{cid}'."))
        ids.add(cid)
        if c.get("layer") not in layer_ids:
            msgs.append(("error", f"[{cid}] layer '{c.get('layer')}' is not defined."))
        for key in ("activity", "from_activity", "to_activity"):
            if key in c and slug(c[key]) not in have_acts:
                msgs.append(("warn", f"[{cid}] activity '{c[key]}' never occurs in the log."))
    for v, spec in norm.get("views", {}).items():
        w = spec.get("layer_weights", {})
        s = sum(w.values())
        if not np.isclose(s, 1.0, atol=0.02):
            msgs.append(("warn", f"View '{v}' layer weights sum to {s:.2f} (will be renormalised)."))
        missing = layer_ids - set(w)
        if missing:
            msgs.append(("warn", f"View '{v}' has no weight for layers {sorted(missing)} (treated as 0)."))
    if not norm.get("views"):
        msgs.append(("warn", "No views defined — a uniform 'Default' view will be used."))
    if not msgs:
        msgs.append(("ok", f"Norm OK: {len(norm['constraints'])} constraints, "
                            f"{len(layer_ids)} layers, {len(norm.get('views', {}))} view(s)."))
    return msgs


def activity_stats(events: pd.DataFrame) -> pd.DataFrame:
    """Frequency table that drives the wizard's suggestions and pickers."""
    n_cases = events["case_id"].nunique()
    g = events.groupby("activity")
    st = pd.DataFrame({
        "events": g.size(),
        "cases": g["case_id"].nunique(),
    })
    st["case_coverage"] = st["cases"] / n_cases
    st["avg_repeats_per_case"] = st["events"] / st["cases"]
    return st.sort_values("events", ascending=False).reset_index()


def common_pairs(events: pd.DataFrame, top=10) -> pd.DataFrame:
    """Most common directly-follows activity pairs with median lag in days."""
    e = events.sort_values(["case_id", "timestamp"])
    same = e["case_id"].eq(e["case_id"].shift(-1))
    nxt_act, nxt_ts = e["activity"].shift(-1), e["timestamp"].shift(-1)
    pairs = pd.DataFrame({"from": e["activity"][same], "to": nxt_act[same],
                          "lag_days": (nxt_ts[same] - e["timestamp"][same]).dt.total_seconds()/86400})
    pairs = pairs[pairs["from"] != pairs["to"]]
    agg = (pairs.groupby(["from", "to"], observed=True)
           .agg(count=("lag_days", "size"), median_lag_days=("lag_days", "median"),
                p90_lag_days=("lag_days", lambda s: s.quantile(0.9)))
           .sort_values("count", ascending=False).reset_index())
    return agg.head(top)


def suggest_norm(events: pd.DataFrame, fc: pd.DataFrame, name="starter norm") -> dict:
    """Propose a starter norm from the log itself. Every suggestion is
    data-grounded and editable; thresholds come from observed quantiles so the
    starter norm is calibrated rather than arbitrary."""
    st = activity_stats(events)
    norm = {"name": name, "version": "0.1-draft",
            "layers": dict(DEFAULT_LAYERS), "constraints": [], "views": {}}
    C = norm["constraints"]

    # L1 presence: the most common *final* activity (closure evidence)
    last_acts = (events.sort_values(["case_id", "timestamp"])
                 .groupby("case_id")["activity"].last().value_counts())
    if len(last_acts):
        closer = last_acts.index[0]
        C.append({"id": f"c_presence_{slug(closer)}", "layer": "L1", "type": "presence",
                  "activity": closer, "min_count": 1,
                  "description": f"Cases should reach '{closer}' "
                                 f"(ends {last_acts.iloc[0]:,} cases)."})

    # L2 lag: top frequent pair with meaningful lag; thresholds = p75 / (p95-p75)
    pairs = common_pairs(events, top=6)
    pairs = pairs[pairs["median_lag_days"] > 0.01]
    if len(pairs):
        p = pairs.iloc[0]
        e = events.sort_values(["case_id", "timestamp"])
        thr = max(round(float(p["median_lag_days"]) * 1.5, 1), 0.5)
        width = max(round(float(p["p90_lag_days"]) - thr, 1), thr)
        C.append({"id": f"c_lag_{slug(p['from'])}_to_{slug(p['to'])}", "layer": "L2",
                  "type": "lag", "from_activity": p["from"], "to_activity": p["to"],
                  "threshold_days": thr, "saturation_days": width,
                  "description": f"'{p['to']}' should follow '{p['from']}' within {thr}d "
                                 f"(observed median {p['median_lag_days']:.1f}d)."})

    # L3 singularity: the most-repeated activity per case
    rep = st[st["avg_repeats_per_case"] > 1.3].head(2)
    for _, r in rep.iterrows():
        allowed = max(int(round(r["avg_repeats_per_case"])), 1) + 1
        C.append({"id": f"c_repeat_{slug(r['activity'])}", "layer": "L3",
                  "type": "singularity", "activity": r["activity"],
                  "allowed": allowed, "saturation": max(allowed, 2),
                  "description": f"Limit repetitions of '{r['activity']}' "
                                 f"(avg {r['avg_repeats_per_case']:.1f}/case)."})

    # L4 exclusion: rare activities whose name suggests exceptions
    exc = st[(st["case_coverage"] < 0.10)
             & st["activity"].str.lower().str.contains("|".join(EXCEPTION_WORDS))]
    for _, r in exc.head(3).iterrows():
        C.append({"id": f"c_excl_{slug(r['activity'])}", "layer": "L4",
                  "type": "exclusion", "activity": r["activity"],
                  "description": f"'{r['activity']}' should not occur "
                                 f"(seen in {r['case_coverage']:.1%} of cases)."})

    # L5 effort: total events per case as a singularity on the busiest activity
    p90 = float(fc["total_events"].quantile(0.90))
    busiest = st.iloc[0]["activity"]
    C.append({"id": f"c_effort_{slug(busiest)}", "layer": "L5", "type": "singularity",
              "activity": busiest,
              "allowed": int(max(fc[f"count__{slug(busiest)}"].quantile(0.90), 2)) if f"count__{slug(busiest)}" in fc else 3,
              "saturation": 3,
              "description": f"High handling effort proxy via repeats of '{busiest}' "
                             f"(p90 case size = {p90:.0f} events)."})

    # default uniform view across layers that actually have constraints
    used_layers = sorted({c["layer"] for c in C})
    w = round(1 / len(used_layers), 3)
    norm["views"]["Default"] = {"layer_weights": {l: w for l in used_layers}}
    return norm
