"""
wise.governance — the governance loop: review → validate → refine → rerun
(Fig. 2, grey band; paper Sec. V-B "validation diagnostics" and Table X).

WISE separates likely *process* issues from likely *measurement* issues
BEFORE any steering conclusion is drawn. Three diagnostics, all computed
from the already-scored data (no new modelling):

1. Right-censoring flag — an invoice-bearing item without Clear Invoice
   that is still active within the final `censoring_window_days` of the
   observation window may simply lack time to finish. Counting it as an
   open-item violation would confuse window truncation with backlog.

2. Event-replication ratio — events per distinct timestamp, per item.
   A ratio above `replication_ratio_flag` (default 2) signals
   extraction-level duplication (BPIC'19 is known to replicate some
   SES-related events), i.e. the "rework" may be a logging artefact.

3. Censoring-robust gap retention — recompute each focus slice's stable
   gap on non-censored items only and report the retained share. If a
   slice's priority collapses without censored items, its hotspot status
   is a window artefact, and demoting it is the *intended* behaviour of
   the loop — not a failure of the method.
"""
import numpy as np
import pandas as pd

from .config import CONFIG


def add_governance_flags(df_case_scores: pd.DataFrame,
                         window_end=None) -> pd.DataFrame:
    """Attach is_right_censored and is_replicated flags per case."""
    d = df_case_scores.copy()
    window_end = window_end or d["case_last_ts"].max()
    horizon = window_end - pd.Timedelta(days=CONFIG["censoring_window_days"])

    has_invoice = (
        d.get("count__record_invoice_receipt", pd.Series(0, index=d.index)).fillna(0)
        + d.get("count__vendor_creates_invoice", pd.Series(0, index=d.index)).fillna(0)
    ) > 0
    has_clear = d.get("count__clear_invoice", pd.Series(0, index=d.index)).fillna(0) > 0
    still_active_late = d["case_last_ts"] >= horizon

    # Right-censored = invoice-bearing, unresolved, and active near the end
    d["is_right_censored"] = has_invoice & (~has_clear) & still_active_late
    d["is_open_invoice"] = has_invoice & (~has_clear)
    d["is_replicated"] = d["event_replication_ratio"] > CONFIG["replication_ratio_flag"]
    d["_window_end"] = window_end
    return d


def gap_retained_excl_censored(d: pd.DataFrame, view: str, slice_mask,
                               gamma: float = None) -> dict:
    """Stable gap of a slice recomputed on non-censored items, as a share
    of the original stable gap (paper Table X, column 'Gap retained')."""
    gamma = CONFIG["shrinkage_gamma"] if gamma is None else gamma
    score = f"score__{view}"

    def stable_gap(frame, mask):
        pop = frame.dropna(subset=[score])
        mu_bar = float(pop[score].mean())
        sl = pop.loc[mask.reindex(pop.index, fill_value=False)]
        if len(sl) == 0:
            return 0.0, 0
        mu_s = float(sl[score].mean())
        n = len(sl)
        mu_shrunk = (n * mu_s + gamma * mu_bar) / (n + gamma)
        return max(mu_bar - mu_shrunk, 0.0), n

    gap_all, n_all = stable_gap(d, slice_mask)
    keep = ~d["is_right_censored"]
    gap_nc, n_nc = stable_gap(d.loc[keep], slice_mask.loc[keep])
    return {
        "stable_gap_all": gap_all,
        "stable_gap_noncensored": gap_nc,
        "gap_retained_share": (gap_nc / gap_all) if gap_all > 0 else np.nan,
        "n_all": n_all, "n_noncensored": n_nc,
    }


def governance_table(d: pd.DataFrame, view: str, focus_slices: list,
                     slice_keys=None) -> pd.DataFrame:
    """Paper Table X — data-quality qualification of the focus slices.

    For each focus slice: right-censored share, replicated-event share,
    gap retained excl. censored, and an auto-generated reading that
    states whether the hotspot survives the data-quality checks.
    """
    slice_keys = slice_keys or CONFIG["slice_keys"]
    rows = []
    for sl in focus_slices:
        mask = pd.Series(True, index=d.index)
        for key, val in zip(slice_keys, sl):
            mask &= d[key] == val
        sub = d.loc[mask]
        rc_share = float(sub["is_right_censored"].mean())
        rep_share = float(sub["is_replicated"].mean())
        ret = gap_retained_excl_censored(d, view, mask)

        # --- auto-generated reading (paper Table X, last column) ---------
        notes = []
        if not np.isnan(ret["gap_retained_share"]) and ret["gap_retained_share"] >= 0.7:
            notes.append("gap largely retained → hotspot is not a window artefact")
        elif not np.isnan(ret["gap_retained_share"]) and ret["gap_retained_share"] >= 0.4:
            notes.append("gap partially retained → interpret with censoring caveat")
        else:
            notes.append("gap collapses without censored items → likely window artefact; demote")
        if rep_share > 0.10:
            notes.append("elevated replication → verify SES/GR logging semantics "
                         "before process conclusions")
        if rc_share <= 0.10 and rep_share <= 0.10:
            notes.append("robust → local review can proceed")
        rows.append({
            "focus_slice": " × ".join(map(str, sl)),
            "cases": int(mask.sum()),
            "right_censored_share": rc_share,
            "replicated_event_share": rep_share,
            "stable_gap_all": ret["stable_gap_all"],
            "stable_gap_excl_censored": ret["stable_gap_noncensored"],
            "gap_retained_excl_censored": ret["gap_retained_share"],
            "reading": "; ".join(notes),
        })
    return pd.DataFrame(rows)


def quarterly_penalty_trend(d: pd.DataFrame, view: str, focus_slices: list,
                            slice_keys=None) -> pd.DataFrame:
    """Mean penalty (1 − S^(p)) by case-start quarter, per focus slice and
    globally — the paper's Fig. 'quarterly trend' (chronic vs drift vs
    truncation)."""
    slice_keys = slice_keys or CONFIG["slice_keys"]
    score = f"score__{view}"
    d = d.dropna(subset=[score]).copy()
    d["penalty"] = 1 - d[score]
    out = d.groupby("case_start_quarter")["penalty"].mean().rename("GLOBAL").to_frame()
    for sl in focus_slices:
        mask = pd.Series(True, index=d.index)
        for key, val in zip(slice_keys, sl):
            mask &= d[key] == val
        out[" × ".join(map(str, sl))] = d.loc[mask].groupby("case_start_quarter")["penalty"].mean()
    return out
