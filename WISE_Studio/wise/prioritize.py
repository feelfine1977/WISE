"""Slice backlog, stable PI, hotspot typology, layer deltas, drill-downs."""
import numpy as np
import pandas as pd


def slice_backlog(scores: pd.DataFrame, view: str, group_cols, gamma: float = 20.0):
    sc = f"score__{view}"
    d = scores.dropna(subset=[sc])
    mu = float(d[sc].mean())
    g = d.groupby(group_cols, dropna=False)
    out = g.agg(cases=("case_id", "size"), slice_mean=(sc, "mean")).reset_index()
    out["global_mean"] = mu
    out["shrunk_mean"] = (out["cases"]*out["slice_mean"] + gamma*mu) / (out["cases"] + gamma)
    out["stable_gap"] = (mu - out["shrunk_mean"]).clip(lower=0)
    out["PI_stable"] = out["cases"] * out["stable_gap"]
    return out.sort_values("PI_stable", ascending=False).reset_index(drop=True)


def pareto_concentration(backlog: pd.DataFrame, thresholds=(0.80, 0.95)):
    pi = backlog["PI_stable"].to_numpy()
    tot = pi.sum()
    if tot <= 0:
        return {t: (0, 0.0) for t in thresholds}, np.array([])
    cum = np.cumsum(pi) / tot
    return {t: (int(np.searchsorted(cum, t) + 1),
                (np.searchsorted(cum, t) + 1) / len(pi)) for t in thresholds}, cum


def classify_hotspots(backlog: pd.DataFrame, top_n=12) -> pd.DataFrame:
    """reservoir = big volume/small gap; severity = small volume/big gap;
    mechanism = the rest. Rank-based so it adapts to any log."""
    top = backlog.head(top_n).copy()
    if len(top) == 0:
        top["hotspot_type"] = []
        return top
    vol_hi = top["cases"] >= top["cases"].quantile(0.75)
    gap_hi = top["stable_gap"] >= top["stable_gap"].quantile(0.75)
    typ = np.where(vol_hi & ~gap_hi, "reservoir",
          np.where(gap_hi & ~vol_hi, "severity", "mechanism"))
    top["hotspot_type"] = typ
    return top


def layer_deltas(scores: pd.DataFrame, view: str, group_cols, norm: dict) -> pd.DataFrame:
    cols = [f"contrib__{view}__{l}" for l in norm["layers"]]
    gm = scores[cols].mean()
    d = scores.groupby(group_cols, dropna=False)[cols].mean() - gm
    d.columns = list(norm["layers"].keys())
    return d


def leading_constraints(scores: pd.DataFrame, mask, norm: dict, top=5) -> pd.DataFrame:
    rows = []
    sub = scores.loc[mask]
    for c in norm["constraints"]:
        v = sub[c["id"]]
        app = v.notna()
        if app.sum() == 0:
            continue
        rows.append({"constraint": c["id"], "layer": c["layer"],
                     "type": c["type"], "fire_rate": float((v > 0).mean()),
                     "penalty_mass": float(v[app].mean()),
                     "description": c.get("description", "")})
    return (pd.DataFrame(rows).sort_values("penalty_mass", ascending=False)
            .head(top).reset_index(drop=True))
