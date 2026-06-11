"""
wise.constraints — bounded violation signals ν_c(σ) ∈ [0,1]  (paper Sec. IV-B).

WISE represents expected behaviour through a small catalogue of
parameterised constraint types. Each constraint maps a case σ to a
*bounded* violation signal: 0 = satisfied, 1 = maximally violated under
the declared cap. Bounded signals make heterogeneous evidence (counts,
time lags, numeric mismatches) comparable and keep aggregation stable.

Paper vocabulary (five types)        Implementation type(s) here
-----------------------------        ----------------------------------------
pres   presence                      presence, presence_any
lag    order/lag with time window    lag                       (order = lag
order  pure ordering (special case)  order                      with δ=∆=0)
sing   singularity (bounded repeats) count_excess, threshold_excess
excl   exclusion (forbidden events)  absence, absence_any
bal    balance (numeric tolerance)   networth_cv proxies, *_with_multiplier

The generic threshold–saturation rule (paper Eq. for sat):

    sat(z; ϑ, W) = 0                 if z ≤ ϑ
                 = min((z−ϑ)/W, 1)   if z > ϑ

Deviations up to the tolerated threshold ϑ incur no penalty, larger
deviations are penalised linearly, and the value caps at 1 once
z ≥ ϑ + W. Example: ϑ=10, W=20 → sat(8)=0, sat(15)=0.25, sat(30)=1.
"""
import numpy as np
import pandas as pd

from .features import slugify


# --------------------------------------------------------------------------
# helpers on the case-feature frame
# --------------------------------------------------------------------------
def count_series(case_df, activity=None, activities_any_of=None):
    """cnt(a, σ) — or the sum over a set of alternative activity labels."""
    if activity is not None:
        col = f"count__{slugify(activity)}"
        return case_df[col] if col in case_df.columns else pd.Series(0, index=case_df.index, dtype=float)
    s = pd.Series(0, index=case_df.index, dtype=float)
    for act in activities_any_of:
        col = f"count__{slugify(act)}"
        if col in case_df.columns:
            s = s + case_df[col].fillna(0)
    return s


def first_ts_series(case_df, first_activity=None, first_any_of=None):
    """t1(a, σ) — first occurrence timestamp (NaT if a never occurs)."""
    if first_activity is not None:
        col = f"first_ts__{slugify(first_activity)}"
        return case_df[col] if col in case_df.columns else pd.Series(pd.NaT, index=case_df.index)
    cols = [f"first_ts__{slugify(a)}" for a in first_any_of
            if f"first_ts__{slugify(a)}" in case_df.columns]
    if not cols:
        return pd.Series(pd.NaT, index=case_df.index)
    return case_df[cols].min(axis=1)


def sat_excess(x, threshold, width):
    """The threshold–saturation rule sat(z; ϑ, W) of paper Sec. IV-B."""
    out = ((x.astype(float) - threshold) / width).clip(lower=0, upper=1)
    return out.where(x.notna(), np.nan)


# --------------------------------------------------------------------------
# constraint evaluation
# --------------------------------------------------------------------------
def evaluate_constraint(case_df: pd.DataFrame, constraint: dict) -> pd.Series:
    """Evaluate one constraint c on every case; returns ν_c(σ) per case.

    NaN encodes "not applicable" — i.e. σ ∉ C_app-domain of c. The
    flow-aware applicability mapping C_app (paper Sec. IV-D) is realised
    by masking on ``applicable_flows`` plus, for lag/order constraints,
    on the existence of both boundary events (the *missing* milestone is
    penalised once in the closure layer L1, never double-counted here).
    """
    typ, p = constraint["type"], constraint["params"]
    app = case_df["flow_type"].isin(constraint["applicable_flows"])
    idx = case_df.index
    viol = pd.Series(np.nan, index=idx, dtype=float)

    if typ in ("presence", "presence_any"):
        # pres: ν = 1 − min(cnt/m, 1)   (indicator-like for m = 1)
        counts = (count_series(case_df, activity=p.get("activity"))
                  if typ == "presence" else count_series(case_df, activities_any_of=p["activities_any_of"]))
        viol = (1 - (counts / p.get("min_count", 1)).clip(upper=1)).where(app, np.nan)

    elif typ in ("absence", "absence_any"):
        # excl: ν = 1[cnt(a,σ) > 0] — any occurrence is maximal violation
        counts = (count_series(case_df, activity=p.get("activity"))
                  if typ == "absence" else count_series(case_df, activities_any_of=p["activities_any_of"]))
        viol = (counts > 0).astype(float).where(app, np.nan)

    elif typ == "order":
        # order: binary lag — ν = 1 iff first b precedes first a
        t_a = (first_ts_series(case_df, first_activity=p["first_activity_a"]) if "first_activity_a" in p
               else first_ts_series(case_df, first_any_of=p["first_any_of_a"]))
        t_b = (first_ts_series(case_df, first_activity=p["first_activity_b"]) if "first_activity_b" in p
               else first_ts_series(case_df, first_any_of=p["first_any_of_b"]))
        applicable = app & t_a.notna() & t_b.notna()
        viol = (t_b < t_a).astype(float).where(applicable, np.nan)

    elif typ == "lag":
        # lag: ν = sat(ℓ; δ, ∆) on ℓ = t1(b) − t1(a) in days
        t_a = (first_ts_series(case_df, first_activity=p["first_activity_a"]) if "first_activity_a" in p
               else first_ts_series(case_df, first_any_of=p["first_any_of_a"]))
        t_b = (first_ts_series(case_df, first_activity=p["first_activity_b"]) if "first_activity_b" in p
               else first_ts_series(case_df, first_any_of=p["first_any_of_b"]))
        applicable = app & t_a.notna() & t_b.notna() & (t_b >= t_a)
        delta_days = (t_b - t_a).dt.total_seconds() / 86400.0
        viol = sat_excess(delta_days, p["threshold_days"], p["saturation_width_days"]).where(applicable, np.nan)

    elif typ == "count_excess":
        # sing: ν = min(max(0, cnt−k)/K, 1)
        counts = (count_series(case_df, activity=p["activity"]) if "activity" in p
                  else count_series(case_df, activities_any_of=p["activities_any_of"]))
        viol = sat_excess(counts.astype(float), p["allowed_count"], p["saturation_width"]).where(app, np.nan)

    elif typ == "threshold_excess":
        # sing on an engineered numeric feature (e.g. manual_touch_count)
        x = case_df[p["feature"]] if p["feature"] in case_df.columns else pd.Series(np.nan, index=idx)
        viol = sat_excess(x.astype(float), p["threshold"], p["saturation_width"]).where(app, np.nan)

    elif typ == "any_of_with_multiplier":
        # bal proxy: occurrence of a correction event scaled by the size of
        # the numeric inconsistency it accompanies (networth_cv ∈ [0,1])
        counts = count_series(case_df, activities_any_of=p["activities_any_of"])
        mult = case_df.get(p["multiplier_feature"], pd.Series(0, index=idx))
        viol = ((counts > 0).astype(float) * mult.clip(lower=0, upper=1)).where(app, np.nan)

    elif typ == "activity_with_multiplier":
        counts = count_series(case_df, activity=p["activity"])
        mult = case_df.get(p["multiplier_feature"], pd.Series(0, index=idx))
        viol = ((counts > 0).astype(float) * mult.clip(lower=0, upper=1)).where(app, np.nan)

    else:
        raise ValueError(f"Unsupported constraint type: {typ}")

    return viol.clip(lower=0, upper=1)


def build_violation_matrix(case_df: pd.DataFrame, norm: dict):
    """Evaluate the whole norm: violation matrix V and applicability mask M.

        V ∈ [0,1]^(n_cases × n_constraints)   ν_c(σ), NaN where N/A
        M ∈ {0,1}^(n_cases × n_constraints)   the C_app mask

    Everything downstream — layer penalties, view scores, layer
    contributions, slice priorities — is a masked weighted average over
    (V, M). Making the two objects explicit keeps the method auditable.
    """
    df_violations = pd.DataFrame({"case_id": case_df["case_id"]})
    for cons in norm["constraints"]:
        df_violations[cons["id"]] = evaluate_constraint(case_df, cons)
    constraint_ids = [c["id"] for c in norm["constraints"]]
    V = df_violations[constraint_ids].to_numpy(dtype=float)
    M = (~np.isnan(V)).astype(float)
    return df_violations, V, M
