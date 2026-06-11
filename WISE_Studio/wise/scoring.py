"""View-specific case scores with exact layer decomposition (generic)."""
import numpy as np
import pandas as pd


def constraint_weights(norm: dict, view: str) -> dict:
    """Raw weight per constraint: layer weight × normalised within-layer weight."""
    lw = norm["views"][view]["layer_weights"]
    by_layer = {}
    for c in norm["constraints"]:
        by_layer.setdefault(c["layer"], []).append(c)
    w = {}
    for layer, cs in by_layer.items():
        tot_b = sum(float(c.get("within_layer_weight", 1.0)) for c in cs)
        for c in cs:
            w[c["id"]] = float(lw.get(layer, 0.0)) * float(c.get("within_layer_weight", 1.0)) / max(tot_b, 1e-12)
    return w


def score_cases(fc: pd.DataFrame, viol: pd.DataFrame, norm: dict) -> pd.DataFrame:
    """Wide frame: features + ν columns + score__<view> + contrib__<view>__<layer>."""
    views = list(norm.get("views") or {"Default": None})
    if not norm.get("views"):
        used = sorted({c["layer"] for c in norm["constraints"]})
        norm["views"] = {"Default": {"layer_weights": {l: 1/len(used) for l in used}}}
        views = ["Default"]
    out = fc.merge(viol, on="case_id")
    cids = [c["id"] for c in norm["constraints"]]
    layer_of = {c["id"]: c["layer"] for c in norm["constraints"]}
    V = out[cids].to_numpy(dtype=float)            # NaN = not applicable
    A = ~np.isnan(V)
    for view in views:
        wmap = constraint_weights(norm, view)
        w = np.array([wmap[c] for c in cids])
        W = A * w                                   # effective raw weights
        denom = W.sum(axis=1)
        ok = denom > 0
        penal = np.where(ok, np.nansum(np.where(A, V, 0) * W, axis=1) / np.where(ok, denom, 1), np.nan)
        out[f"score__{view}"] = 1 - penal
        # exact per-layer decomposition of the penalty
        for layer in norm["layers"]:
            sel = np.array([layer_of[c] == layer for c in cids])
            contrib = np.where(ok, np.nansum(np.where(A, V, 0) * (W * sel), axis=1)
                               / np.where(ok, denom, 1), np.nan)
            out[f"contrib__{view}__{layer}"] = contrib
    return out
