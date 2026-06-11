"""
wise.prioritization — Phase 3: prioritise documents and ownership slices
(Fig. 2, purple band; paper Sec. IV-E).

Priority Index (the heart of slice-first decision support):

    PI^(p)_s = n_s · ( μ̄^(p) − μ_s^(p) )_+          scale × underperformance

A slice rises in the backlog when a meaningful score gap and sufficient
scale *coincide* — a tiny slice with a large gap can still rank below a
huge slice with a moderate gap ("act where deviations occur at scale").

Shrinkage stabilisation for small slices (paper "optional stabilisation"):

    μ̃_s = n_s/(n_s+γ) · μ_s + γ/(n_s+γ) · μ̄
    PĨ_s = v_s · ( μ̄ − μ̃_s )_+        v_s = n_s (volume) or E_s (exposure)

Small slices are pulled toward the global mean, damping noisy gaps; for
large n_s the stabilised mean approaches the observed slice mean, and
γ = 0 recovers the basic PI.

Hotspot typology (paper Sec. V, operational labels, not literature terms):
    reservoir  — big n_s, moderate gap  (volume carries the priority)
    severity   — small n_s, large gap   (intensity carries the priority)
    mechanism  — in between, one layer/constraint dominates the profile
"""
import numpy as np
import pandas as pd

from .config import CONFIG


def slice_backlog(df_case_scores: pd.DataFrame, view: str, group_cols,
                  gamma: float = None, exposure_col: str = None) -> pd.DataFrame:
    """Aggregate case scores into a ranked slice backlog for one view.

    Columns: cases n_s, slice mean μ_s, raw gap, stabilised mean μ̃_s,
    stable gap, PI (raw) and stable PI — ranked by stable PI.
    """
    gamma = CONFIG["shrinkage_gamma"] if gamma is None else gamma
    score_col = f"score__{view}"
    d = df_case_scores.dropna(subset=[score_col])
    global_mean = float(d[score_col].mean())

    g = d.groupby(group_cols, dropna=False)
    out = g.agg(cases=("case_id", "size"), slice_mean=(score_col, "mean")).reset_index()
    out["global_mean"] = global_mean
    out["gap_raw"] = (global_mean - out["slice_mean"]).clip(lower=0)
    out["shrunk_mean"] = (out["cases"] * out["slice_mean"] + gamma * global_mean) / (out["cases"] + gamma)
    out["stable_gap"] = (global_mean - out["shrunk_mean"]).clip(lower=0)
    out["PI_raw"] = out["cases"] * out["gap_raw"]
    out["PI_stable"] = out["cases"] * out["stable_gap"]
    if exposure_col is not None and exposure_col in d.columns:
        exp_sum = g[exposure_col].sum().reset_index(drop=True)
        out["exposure"] = exp_sum
        out["PI_stable_exposure"] = out["exposure"] * out["stable_gap"]
    return out.sort_values("PI_stable", ascending=False).reset_index(drop=True)


def document_backlog(df_case_scores: pd.DataFrame, view: str,
                     gamma: float = None) -> pd.DataFrame:
    """PO-header roll-up — the governance unit for the Q3.1 stand-out question."""
    return slice_backlog(df_case_scores, view, ["purchasing_document"], gamma=gamma)


def pareto_concentration(backlog: pd.DataFrame, thresholds=(0.80, 0.95)):
    """How many top-ranked units capture X% of total stable-PI mass?"""
    pi = backlog["PI_stable"].to_numpy()
    total = pi.sum()
    if total <= 0:
        return {t: (0, 0.0) for t in thresholds}, np.array([])
    cum = np.cumsum(pi) / total
    res = {}
    for t in thresholds:
        k = int(np.searchsorted(cum, t) + 1)
        res[t] = (k, k / len(pi))
    return res, cum


def jaccard_topk(backlogs: dict, k: int = None) -> pd.DataFrame:
    """Pairwise Jaccard overlap of the top-k shortlists across views.

    Population-level score correlations can stay high while shortlists
    diverge — shortlists live in the extreme tail (paper finding 6).
    """
    k = CONFIG["top_k_documents"] if k is None else k
    views = list(backlogs.keys())
    tops = {v: set(backlogs[v].head(k).iloc[:, 0]) for v in views}
    mat = pd.DataFrame(index=views, columns=views, dtype=float)
    for a in views:
        for b in views:
            inter = len(tops[a] & tops[b])
            union = len(tops[a] | tops[b])
            mat.loc[a, b] = inter / union if union else np.nan
    return mat


def classify_hotspots(backlog: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Attach the operational hotspot typology to the top of a backlog.

    Thresholds are quantile-based on the top-n *positive-PI* slices: a
    slice is a *reservoir* if its volume is in the upper third and its
    gap in the lower third (of the top set), a *severity* hotspot in the
    mirrored situation, and a *mechanism* hotspot otherwise. Slices with
    zero stable PI carry no improvement potential and are excluded — they
    are kept in the backlog display but never typed as hotspots. The
    labels guide the type of follow-up, not the maths.
    """
    pos = backlog[backlog["PI_stable"] > 0]
    top = pos.head(top_n).copy()
    if len(top) == 0:
        out = backlog.head(top_n).copy()
        out["hotspot_type"] = "—"
        return out
    # Rank-based typing on the positive-PI set: the slice combining the
    # highest volume rank with the lowest gap rank is the *reservoir*; the
    # mirrored extreme is the *severity* hotspot; everything else is a
    # *mechanism* hotspot. Ranks (not absolute thresholds) keep the labels
    # meaningful for small positive sets.
    vr = top["cases"].rank(method="average")
    gr = top["stable_gap"].rank(method="average")
    top["hotspot_type"] = "mechanism"
    if len(top) >= 2:
        res_score = vr - gr
        sev_score = gr - vr
        i_res = res_score.sort_values(ascending=False).index[0]
        # severity tiebreak: prefer the smallest slice
        sev_order = top.assign(_s=sev_score).sort_values(
            ["_s", "cases"], ascending=[False, True]).index
        i_sev = next(i for i in sev_order if i != i_res)
        if res_score.loc[i_res] > 0:
            top.loc[i_res, "hotspot_type"] = "reservoir"
        if sev_score.loc[i_sev] > 0:
            top.loc[i_sev, "hotspot_type"] = "severity"
    return top


def slice_layer_deltas(df_case_scores: pd.DataFrame, view: str, group_cols,
                       norm: dict) -> pd.DataFrame:
    """Mean layer contribution Δ_λ^(p) per slice MINUS the global mean.

    Positive cells mark deviation mechanisms more pronounced in the slice
    than in the log overall — the paper's Fig. 'layer-delta profiles'.
    """
    layer_ids = list(norm["layers"].keys())
    cols = [f"contrib__{view}__{l}" for l in layer_ids]
    global_mean = df_case_scores[cols].mean()
    g = df_case_scores.groupby(group_cols, dropna=False)[cols].mean()
    deltas = g - global_mean
    deltas.columns = layer_ids
    return deltas


def leading_constraint(df_case_scores: pd.DataFrame, slice_mask, layer_id: str,
                       norm: dict) -> pd.Series:
    """Within a slice and a layer, which constraint carries the most
    violation mass? (mean ν_c over applicable cases × within-layer weight)"""
    lcs = [c for c in norm["constraints"] if c["layer_id"] == layer_id]
    total_b = sum(c.get("within_layer_weight", 1.0) for c in lcs)
    scores = {}
    sub = df_case_scores.loc[slice_mask]
    for c in lcs:
        v = sub[c["id"]]
        scores[c["id"]] = float(v.mean(skipna=True) or 0) * c.get("within_layer_weight", 1.0) / total_b
    return pd.Series(scores).sort_values(ascending=False)
