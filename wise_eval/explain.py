"""
wise_eval.explain — root-cause and explanatory analysis on top of WISE scores.

WISE answers *where* deviation concentrates and *which layer* dominates a slice.
This module adds the next analytical step the paper's Future Work calls for
("stronger links to root-cause analysis"): methods that help explain *why* a
hotspot underperforms and *what co-occurs* with the deviation, while staying
descriptive (no causal claims).

Each function takes the wide `scores` frame (one row per PO item, produced by
wise_eval.scoring.assemble_case_scores) and returns a tidy DataFrame/figure so
it can be reused outside the notebook.

Methods provided
----------------
  constraint_prevalence      : per-constraint firing rate + mean severity (global or slice)
  constraint_cooccurrence    : Jaccard / lift between violated constraints (which problems travel together)
  contrastive_drivers        : slice-vs-rest standardized mean differences (Cohen's d) over features+constraints
  surrogate_tree_rules       : shallow decision tree → human-readable rules for "what predicts a low score"
  permutation_importance_for_score : model-agnostic feature importance for the case score
  resource_handoff_matrix    : resource→resource handover counts within a slice (segregation-of-duties / churn)
  variant_severity_table     : top trace variants in a slice ranked by mean penalty (which behaviours are worst)
  spend_value_phase_profile  : value-phase breakdown of penalty mass (where in the money flow it sits)
"""
from itertools import combinations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _slice_mask(scores, slice_keys, slice_vals):
    m = pd.Series(True, index=scores.index)
    for k, v in zip(slice_keys, slice_vals):
        m &= (scores[k] == v)
    return m


def _constraint_ids(norm):
    return [c["id"] for c in norm["constraints"]]


def _layer_of(norm):
    return {c["id"]: c["layer_id"] for c in norm["constraints"]}


# --------------------------------------------------------------------------
# 1. Constraint prevalence — which individual checks fire, and how hard
# --------------------------------------------------------------------------
def constraint_prevalence(scores, norm, mask=None):
    """Per-constraint firing rate and mean severity over applicable cases.

    Returns a frame sorted by 'penalty_mass' = fire_rate × mean_severity,
    the single best proxy for "which constraint contributes most deviation".
    Works globally (mask=None) or for one slice (boolean mask).
    """
    sub = scores if mask is None else scores.loc[mask]
    lay = _layer_of(norm)
    rows = []
    for cid in _constraint_ids(norm):
        if cid not in sub.columns:
            continue
        v = sub[cid]
        applicable = v.notna()
        n_app = int(applicable.sum())
        if n_app == 0:
            continue
        fired = (v > 0) & applicable
        rows.append({
            "constraint": cid,
            "layer": lay[cid],
            "applicable_cases": n_app,
            "fire_rate": float(fired.mean()),
            "mean_severity": float(v[applicable].mean()),
            "mean_severity_when_fired": float(v[fired].mean()) if fired.any() else 0.0,
            "penalty_mass": float(v[applicable].mean()),  # = fire_rate-weighted severity
        })
    return (pd.DataFrame(rows)
            .sort_values("penalty_mass", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------
# 2. Constraint co-occurrence — which problems travel together
# --------------------------------------------------------------------------
def constraint_cooccurrence(scores, norm, mask=None, min_support=200):
    """Pairwise co-occurrence of violated constraints.

    For every pair of constraints that both fire (>0) we report support,
    Jaccard overlap, and lift = P(A&B)/(P(A)P(B)). Lift > 1 means the two
    deviations co-occur more than chance — a structural hint that they share
    a mechanism (e.g. fragmentation + late clearing).
    """
    sub = scores if mask is None else scores.loc[mask]
    cids = [c for c in _constraint_ids(norm) if c in sub.columns]
    fired = {c: (sub[c] > 0).fillna(False) for c in cids}
    n = len(sub)
    base = {c: float(fired[c].mean()) for c in cids}
    rows = []
    for a, b in combinations(cids, 2):
        both = (fired[a] & fired[b]).sum()
        if both < min_support:
            continue
        either = (fired[a] | fired[b]).sum()
        pa, pb = base[a], base[b]
        lift = (both / n) / (pa * pb) if pa > 0 and pb > 0 else np.nan
        rows.append({"constraint_a": a, "constraint_b": b,
                     "support_both": int(both),
                     "jaccard": both / either if either else 0.0,
                     "lift": float(lift)})
    if not rows:
        return pd.DataFrame(columns=["constraint_a", "constraint_b",
                                     "support_both", "jaccard", "lift"])
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Contrastive drivers — slice vs rest, standardized mean difference
# --------------------------------------------------------------------------
def contrastive_drivers(scores, norm, mask, feature_cols=None, top=15):
    """Cohen's d of each feature/constraint: in-slice vs rest-of-log.

    This is the workhorse of contrastive root-cause analysis: it ranks which
    measurable quantities most separate the hotspot from everything else.
    Positive d = higher in the slice. Robust to scale (standardized).
    """
    in_grp = scores.loc[mask]
    out_grp = scores.loc[~mask]
    if feature_cols is None:
        feats = [c for c in _constraint_ids(norm) if c in scores.columns]
        extra = ["manual_share", "manual_touch_count", "distinct_human_resources",
                 "total_events", "event_replication_ratio", "networth_cv",
                 "exposure_norm", "distinct_timestamps"]
        feats += [c for c in extra if c in scores.columns]
        feats += [c for c in scores.columns if c.startswith("count__")]
    else:
        feats = [c for c in feature_cols if c in scores.columns]
    rows = []
    for f in feats:
        a = pd.to_numeric(in_grp[f], errors="coerce").dropna()
        b = pd.to_numeric(out_grp[f], errors="coerce").dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        na, nb = len(a), len(b)
        sp = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / max(na + nb - 2, 1))
        d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
        rows.append({"feature": f, "in_mean": float(a.mean()),
                     "rest_mean": float(b.mean()), "cohens_d": float(d)})
    out = pd.DataFrame(rows)
    out["abs_d"] = out["cohens_d"].abs()
    return out.sort_values("abs_d", ascending=False).head(top).reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Surrogate decision tree — readable rules for a low score
# --------------------------------------------------------------------------
def surrogate_tree_rules(scores, view, feature_cols=None, max_depth=3,
                         low_quantile=0.10):
    """Fit a shallow decision tree that predicts 'is this case a low scorer?'
    and return the tree + extracted rules as plain text.

    Interpretable surrogate (paper's XAI angle): the tree approximates the
    score with a handful of human-readable thresholds, e.g.
    'count__record_goods_receipt > 4 AND manual_share > 0.6 → low score'.
    Returns (clf, rules_text, feature_names).
    """
    from sklearn.tree import DecisionTreeClassifier, export_text

    score_col = f"score__{view}"
    d = scores.dropna(subset=[score_col]).copy()
    thr = d[score_col].quantile(low_quantile)
    y = (d[score_col] <= thr).astype(int)

    if feature_cols is None:
        feature_cols = ([c for c in d.columns if c.startswith("count__")] +
                        [c for c in ["manual_share", "manual_touch_count",
                                     "distinct_human_resources", "total_events",
                                     "event_replication_ratio", "networth_cv",
                                     "exposure_norm"] if c in d.columns])
    X = d[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=max(50, len(d)//200),
                                 class_weight="balanced", random_state=0)
    clf.fit(X, y)
    rules = export_text(clf, feature_names=list(feature_cols), max_depth=max_depth)
    return clf, rules, feature_cols


def permutation_importance_for_score(scores, view, feature_cols=None,
                                     sample=20000, n_repeats=5):
    """Model-agnostic permutation importance for the case score.

    Fits a small gradient-boosted regressor S^(p) ~ features and measures how
    much each feature matters by the drop in R² when it is shuffled. Answers:
    'across the whole log, which raw quantities most drive the score?'
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    score_col = f"score__{view}"
    d = scores.dropna(subset=[score_col])
    if len(d) > sample:
        d = d.sample(sample, random_state=0)
    if feature_cols is None:
        feature_cols = ([c for c in d.columns if c.startswith("count__")] +
                        [c for c in ["manual_share", "distinct_human_resources",
                                     "total_events", "event_replication_ratio",
                                     "networth_cv", "exposure_norm"] if c in d.columns])
    X = d[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = d[score_col].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    reg = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.1,
                                        max_iter=200, random_state=0).fit(Xtr, ytr)
    r2 = reg.score(Xte, yte)
    pi = permutation_importance(reg, Xte, yte, n_repeats=n_repeats,
                                random_state=0, n_jobs=1)
    out = (pd.DataFrame({"feature": feature_cols,
                         "importance": pi.importances_mean,
                         "std": pi.importances_std})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    return out, float(r2)


# --------------------------------------------------------------------------
# 5. Resource hand-off matrix — SoD / coordination overhead
# --------------------------------------------------------------------------
def resource_handoff_matrix(df, mask_case_ids, top_resources=12):
    """Resource→resource handover counts within a set of cases.

    A hand-off is two consecutive events in the same case performed by
    different resources. Dense off-diagonal mass = many human hand-offs
    (coordination overhead / SoD surface). Returns a square DataFrame.
    """
    sub = df[df["case_id"].isin(mask_case_ids)].sort_values(
        ["case_id", "event time:timestamp"])
    res = sub["event org:resource"].astype(str)
    prev = res.shift(1)
    same_case = sub["case_id"].eq(sub["case_id"].shift(1))
    handoffs = pd.DataFrame({"frm": prev[same_case], "to": res[same_case]})
    handoffs = handoffs[handoffs["frm"] != handoffs["to"]]
    top = handoffs["frm"].value_counts().head(top_resources).index.union(
        handoffs["to"].value_counts().head(top_resources).index)
    h = handoffs[handoffs["frm"].isin(top) & handoffs["to"].isin(top)]
    return (h.groupby(["frm", "to"]).size().unstack(fill_value=0)
            .reindex(index=top, columns=top, fill_value=0))


# --------------------------------------------------------------------------
# 6. Variant severity — which behaviours are worst inside a slice
# --------------------------------------------------------------------------
def variant_severity_table(df, scores, view, mask, top=12):
    """Top trace variants in a slice, ranked by mean penalty.

    A variant is the ordered sequence of distinct activities. This links the
    PI hotspot back to concrete behavioural patterns — the bridge from
    'this slice is bad' to 'these execution sequences are bad'.
    """
    case_ids = scores.loc[mask, "case_id"]
    seqs = (df[df["case_id"].isin(case_ids)]
            .sort_values(["case_id", "event time:timestamp"])
            .groupby("case_id")["event_activity"]
            .apply(lambda s: " → ".join(pd.unique(s))))
    pen = (1 - scores.set_index("case_id")[f"score__{view}"]).reindex(seqs.index)
    vt = pd.DataFrame({"variant": seqs, "penalty": pen})
    agg = (vt.groupby("variant")
           .agg(cases=("penalty", "size"), mean_penalty=("penalty", "mean"))
           .reset_index())
    agg["penalty_mass"] = agg["cases"] * agg["mean_penalty"]
    return agg.sort_values("penalty_mass", ascending=False).head(top).reset_index(drop=True)


# --------------------------------------------------------------------------
# 7. Spend / value-phase profile
# --------------------------------------------------------------------------
def value_phase_profile(scores, view, mask, exposure_col="exposure_norm",
                        n_bins=5):
    """Penalty mass by exposure (value) quantile inside a slice.

    Splits the slice's items into value bands and reports how penalty mass
    distributes across them. Answers the Q3.3 value angle: is the deviation
    in the high-value tail (financially material) or in low-value noise?
    """
    sub = scores.loc[mask].copy()
    if exposure_col not in sub.columns:
        return pd.DataFrame()
    sub["penalty"] = 1 - sub[f"score__{view}"]
    sub = sub.dropna(subset=["penalty", exposure_col])
    if len(sub) < n_bins:
        return pd.DataFrame()
    sub["value_band"] = pd.qcut(sub[exposure_col].rank(method="first"),
                                n_bins, labels=[f"Q{i+1}" for i in range(n_bins)])
    out = (sub.groupby("value_band")
           .agg(cases=("penalty", "size"), mean_penalty=("penalty", "mean"),
                total_exposure=(exposure_col, "sum"))
           .assign(penalty_mass=lambda t: t["cases"] * t["mean_penalty"]))
    out["penalty_mass_share"] = out["penalty_mass"] / out["penalty_mass"].sum()
    return out.reset_index()
