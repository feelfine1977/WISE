"""
wise.viz — all evaluation figures of paper Section V, including the ones
the paper leaves as placeholders ("ECDF placeholder", "Vendor Pareto
placeholder", "Quarterly trend placeholder").

Figure map (paper → function):
  Fig. 4/11  document Pareto curve                → fig_document_pareto
  Fig. 5/12  risk matrix (scale × severity)       → fig_risk_matrix
  Fig. 6/13  layer-delta profiles                 → fig_layer_deltas
  Fig. 7     ECDF of raw IR→CI lag (placeholder)  → fig_lag_ecdf
  Fig. 8/14  ownership heatmap                    → fig_ownership_heatmap
  Fig. 9     within-hotspot vendor Pareto (ph.)   → fig_vendor_pareto
  Fig. 10    quarterly penalty trend (ph.)        → fig_quarterly_trend
  extra      layer-weight heatmap (Table VIII)    → fig_layer_weights
  extra      score distributions per view         → fig_score_distributions
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import CONFIG

def _fig_dir():
    """Read at call time so notebooks can redirect via CONFIG['figures_dir']."""
    p = Path(CONFIG["figures_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p

VIEW_COLORS = {"Finance": "#1f77b4", "Logistics": "#2ca02c",
               "Compliance": "#9467bd", "Automation": "#d62728"}


def _save(fig, name):
    fig.savefig(_fig_dir() / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig


def fig_layer_weights(layer_table: pd.DataFrame, name="fig_layer_weights"):
    """Heatmap of Table VIII: layer weights a_λ^(p) per view, focal layers
    visible as the dark cells of each column."""
    views = [c for c in layer_table.columns if c != "layer_name"]
    mat = layer_table[views].astype(float)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(mat.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=mat.values.max())
    ax.set_xticks(range(len(views)), views)
    ax.set_yticks(range(len(mat)), [f"{i} {n}" for i, n in
                                    zip(mat.index, layer_table["layer_name"])], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.20 else "black",
                    fontweight="bold" if v == mat[views[j]].max() else "normal")
    ax.set_title("Layer-level role weights $a_\\lambda^{(p)}$ (paper Table VIII)\n"
                 "bold = focal layer of the view; each column sums to 1")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _save(fig, name)


def fig_score_distributions(df_case_scores, views, name="fig_score_distributions"):
    """Per-view case score distributions with global baselines μ̄^(p)."""
    fig, axes = plt.subplots(1, len(views), figsize=(4 * len(views), 3.2), sharey=True)
    for ax, v in zip(np.atleast_1d(axes), views):
        s = df_case_scores[f"score__{v}"].dropna()
        ax.hist(s, bins=40, color=VIEW_COLORS.get(v, "grey"), alpha=0.75)
        ax.axvline(s.mean(), color="black", ls="--", lw=1.2)
        ax.set_title(f"{v}\n$\\bar\\mu$ = {s.mean():.3f}")
        ax.set_xlabel("case score $S^{(p)}(\\sigma)$")
    np.atleast_1d(axes)[0].set_ylabel("cases")
    fig.suptitle("Phase 2 output — view-specific case scores on the same shared norm", y=1.04)
    fig.tight_layout()
    return _save(fig, name)


def fig_document_pareto(cum_by_view: dict, threshold=0.80, name="fig_document_pareto"):
    """Paper Fig. 4/11 — cumulative stable-PI mass over top-k PO headers."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for v, cum in cum_by_view.items():
        if len(cum) == 0:
            continue
        x = np.arange(1, len(cum) + 1)
        ax.plot(x, cum, label=v, color=VIEW_COLORS.get(v, None), lw=1.6)
        k80 = int(np.searchsorted(cum, threshold) + 1)
        ax.scatter([k80], [cum[k80 - 1]], color=VIEW_COLORS.get(v, None), zorder=5, s=18)
    ax.axhline(threshold, color="grey", ls="--", lw=1)
    ax.text(0.99, threshold + 0.012, f"{threshold:.0%} of priority mass",
            transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color="grey")
    ax.set_xlabel("top-k purchasing documents (PO headers), ranked by stable PI")
    ax.set_ylabel("cumulative share of stable PI")
    ax.set_title("Q3.1 — Backlog concentration: a few documents carry most priority mass")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def fig_risk_matrix(backlog_typed: pd.DataFrame, slice_keys, view,
                    name="fig_risk_matrix"):
    """Paper Fig. 5/12 — scale (x) × stabilised underperformance (y),
    stable PI as bubble size/colour; hotspot types occupy distinct regions."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    markers = {"reservoir": "s", "mechanism": "o", "severity": "^"}
    for typ, sub in backlog_typed.groupby("hotspot_type"):
        sc = ax.scatter(sub["cases"], sub["stable_gap"],
                        s=40 + 220 * sub["PI_stable"] / backlog_typed["PI_stable"].max(),
                        c=sub["PI_stable"], cmap="YlOrRd", marker=markers.get(typ, "o"),
                        vmin=0, vmax=backlog_typed["PI_stable"].max(),
                        edgecolor="k", linewidth=0.6, label=typ)
    for _, r in backlog_typed.head(6).iterrows():
        label = " × ".join(str(r[k]) for k in slice_keys)
        ax.annotate(label, (r["cases"], r["stable_gap"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("slice volume $n_s$ (log scale)")
    ax.set_ylabel("stabilised underperformance gap")
    ax.set_title(f"Q3.2 — {view} risk matrix: reservoirs far right at low gap,\n"
                 "severity hotspots far up at low volume, mechanisms in between")
    ax.legend(title="hotspot type", fontsize=8)
    fig.colorbar(sc, ax=ax, label="stable PI")
    fig.tight_layout()
    return _save(fig, name)


def fig_layer_deltas(deltas: pd.DataFrame, norm, view, name="fig_layer_deltas"):
    """Paper Fig. 6/13 — layer-delta profiles of focus slices vs global.

    Positive bars: mechanism MORE pronounced in the slice than overall."""
    layer_names = [norm["layers"][l]["name"] for l in deltas.columns]
    n = len(deltas)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.6), sharey=True)
    for ax, (idx, row) in zip(np.atleast_1d(axes), deltas.iterrows()):
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in row.values]
        ax.barh(range(len(row)), row.values, color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(row)), layer_names, fontsize=7)
        title = idx if isinstance(idx, str) else " × ".join(map(str, idx))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Δ layer contribution vs global")
    fig.suptitle(f"Q3.2 — Layer-delta profiles ({view} view): which deviation mechanism "
                 "dominates each focus slice", y=1.05)
    fig.tight_layout()
    return _save(fig, name)


def fig_lag_ecdf(lag_in, lag_out, slice_label, name="fig_lag_ecdf"):
    """Paper Fig. 7 (placeholder in the paper) — ECDF of the raw IR→CI lag,
    in-slice vs rest of log. Returns (fig, verdict_dict).

    Diagnostic logic: a body-wide rightward shift ⇒ systemic queue problem
    (levers: SLA management, escalation, workload rebalancing); a heavy
    tail on a normal body ⇒ stuck subpopulation (lever: exception triage).
    """
    lag_in = pd.Series(lag_in).dropna()
    lag_out = pd.Series(lag_out).dropna()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s, lab, col in [(lag_in, f"{slice_label} (n={len(lag_in)})", "#d62728"),
                        (lag_out, f"rest of log (n={len(lag_out)})", "#1f77b4")]:
        x = np.sort(s.values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", label=lab, color=col, lw=1.6)
        ax.axvline(np.median(x), color=col, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("raw invoice → clear lag (days, log axis)")
    ax.set_ylabel("ECDF")
    med_in, med_out = float(lag_in.median()), float(lag_out.median())
    p90_in, p90_out = float(lag_in.quantile(0.9)), float(lag_out.quantile(0.9))
    tail_in = p90_in / max(med_in, 1e-9)
    tail_out = p90_out / max(med_out, 1e-9)
    shift = med_in / max(med_out, 1e-9)
    verdict = ("body-wide shift → systemic queue problem (SLA/escalation/"
               "workload rebalancing)" if shift > 1.5 and tail_in <= tail_out * 1.3
               else "heavy tail → stuck subpopulation (targeted exception triage)"
               if tail_in > tail_out * 1.3 and shift <= 1.5
               else "shift AND heavy tail → both queue management and exception triage")
    ax.set_title(f"Q3.2 — IR→CI lag, {slice_label} vs rest\n"
                 f"median {med_in:.0f}d vs {med_out:.0f}d (×{shift:.1f}); "
                 f"p90/median {tail_in:.1f} vs {tail_out:.1f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name), {"median_in": med_in, "median_out": med_out,
                              "shift_ratio": shift, "tail_in": tail_in,
                              "tail_out": tail_out, "verdict": verdict}


def fig_ownership_heatmap(backlog, slice_keys, view, name="fig_ownership_heatmap"):
    """Paper Fig. 8/14 — PI concentration by company × spend area,
    relative to the average cell. Positive cells carry above-average mass."""
    pivot = backlog.pivot_table(index=slice_keys[0], columns=slice_keys[1],
                                values="PI_stable", aggfunc="sum", fill_value=0.0)
    rel = pivot - pivot.values.mean()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    vmax = np.abs(rel.values).max() or 1.0
    im = ax.imshow(rel.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(rel.shape[1]), rel.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(rel.shape[0]), rel.index, fontsize=8)
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            ax.text(j, i, f"{rel.values[i, j]:.0f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title(f"Q3.3 — {view} stable-PI concentration by {slice_keys[0]} × {slice_keys[1]}\n"
                 "relative to the average cell (red = above-average priority mass)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="PI mass − average cell")
    fig.tight_layout()
    return _save(fig, name)


def fig_vendor_pareto(vendor_penalty: pd.Series, hotspot_label,
                      threshold=0.80, name="fig_vendor_pareto"):
    """Paper Fig. 9 (placeholder in the paper) — penalty mass by vendor
    inside a hotspot (bars) and cumulative share (line).
    Returns (fig, k_of_K dict)."""
    s = vendor_penalty.sort_values(ascending=False)
    cum = s.cumsum() / s.sum()
    k80 = int(np.searchsorted(cum.values, threshold) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    show = min(len(s), 25)
    ax.bar(range(show), s.values[:show], color="#d62728", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(range(show), cum.values[:show], color="black", marker="o", ms=3, lw=1.4)
    ax2.axhline(threshold, color="grey", ls="--", lw=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("cumulative share")
    ax.set_xticks(range(show), s.index[:show], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("penalty mass")
    ax.set_title(f"Q3.3 — Within-hotspot vendor Pareto: {hotspot_label}\n"
                 f"{k80} of {len(s)} vendors carry {threshold:.0%} of the penalty mass")
    fig.tight_layout()
    return _save(fig, name), {"k": k80, "K": len(s), "threshold": threshold}


def fig_quarterly_trend(trend: pd.DataFrame, censor_quarters, view,
                        name="fig_quarterly_trend"):
    """Paper Fig. 10 (placeholder in the paper) — mean penalty by case-start
    quarter, focus slices vs global, censoring-affected quarters shaded."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    quarters = list(trend.index)
    for col in trend.columns:
        style = dict(lw=2.4, color="black") if col == "GLOBAL" else dict(lw=1.6)
        ax.plot(quarters, trend[col].values, marker="o", ms=3, label=col, **style)
    for q in censor_quarters:
        if q in quarters:
            i = quarters.index(q)
            ax.axvspan(i - 0.5, i + 0.5, color="grey", alpha=0.18)
    ax.set_xticks(range(len(quarters)), quarters, rotation=30, fontsize=8)
    ax.set_ylabel("mean penalty  $1 - S^{(p)}$")
    ax.set_title(f"Governance loop — mean {view} penalty by case-start quarter\n"
                 "(shaded = censoring-affected final quarters of the window)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


# ===========================================================================
# Root-cause / explanatory figures (wise_eval.explain companions)
# ===========================================================================
def fig_constraint_prevalence(prev_df, title, name="fig_constraint_prevalence",
                              top=18):
    """Horizontal bar of penalty mass per constraint, coloured by layer."""
    d = prev_df.head(top).iloc[::-1]
    layers = sorted(d["layer"].unique())
    cmap = plt.get_cmap("tab10")
    lcol = {l: cmap(i % 10) for i, l in enumerate(layers)}
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.34 * len(d))))
    ax.barh(range(len(d)), d["penalty_mass"], color=[lcol[l] for l in d["layer"]])
    ax.set_yticks(range(len(d)), d["constraint"], fontsize=7)
    ax.set_xlabel("penalty mass  (fire-rate × mean severity over applicable cases)")
    ax.set_title(title, fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=lcol[l]) for l in layers]
    ax.legend(handles, layers, fontsize=7, loc="lower right", title="layer")
    fig.tight_layout()
    return _save(fig, name)


def fig_cooccurrence_heatmap(scores, norm, mask, title,
                             name="fig_cooccurrence", min_fire=0.01):
    """Lift heatmap of constraint co-occurrence within a slice."""
    cids = [c["id"] for c in norm["constraints"] if c["id"] in scores.columns]
    sub = scores if mask is None else scores.loc[mask]
    fired = {c: (sub[c] > 0).fillna(False) for c in cids}
    keep = [c for c in cids if fired[c].mean() >= min_fire]
    n = len(sub)
    M = np.full((len(keep), len(keep)), np.nan)
    for i, a in enumerate(keep):
        pa = fired[a].mean()
        for j, b in enumerate(keep):
            pb = fired[b].mean()
            both = (fired[a] & fired[b]).mean()
            M[i, j] = (both) / (pa * pb) if pa > 0 and pb > 0 else np.nan
    fig, ax = plt.subplots(figsize=(0.5 * len(keep) + 3, 0.5 * len(keep) + 2))
    im = ax.imshow(np.clip(M, 0, 3), cmap="RdBu_r", vmin=0, vmax=2)
    ax.set_xticks(range(len(keep)), keep, rotation=90, fontsize=6)
    ax.set_yticks(range(len(keep)), keep, fontsize=6)
    ax.set_title(title + "\n(lift > 1 = co-occur more than chance)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="lift")
    fig.tight_layout()
    return _save(fig, name)


def fig_contrastive_drivers(contrast_df, title, name="fig_contrastive_drivers"):
    """Diverging bar of Cohen's d: what separates the slice from the rest."""
    d = contrast_df.iloc[::-1]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in d["cohens_d"]]
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.34 * len(d))))
    ax.barh(range(len(d)), d["cohens_d"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    for x in (-0.8, -0.5, 0.5, 0.8):
        ax.axvline(x, color="grey", ls=":", lw=0.6)
    ax.set_yticks(range(len(d)), d["feature"], fontsize=7)
    ax.set_xlabel("Cohen's d  (in-slice − rest; |d|>0.5 medium, >0.8 large)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, name)


def fig_permutation_importance(imp_df, r2, view, name="fig_perm_importance", top=15):
    """Permutation importance of features for the case score."""
    d = imp_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.34 * len(d))))
    ax.barh(range(len(d)), d["importance"], xerr=d["std"], color="#2ca02c",
            ecolor="grey", capsize=2)
    ax.set_yticks(range(len(d)), d["feature"], fontsize=7)
    ax.set_xlabel("permutation importance (drop in R² when shuffled)")
    ax.set_title(f"What drives the {view} case score? (surrogate R² = {r2:.2f})",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, name)


def fig_handoff_matrix(H, title, name="fig_handoff_matrix"):
    """Resource→resource hand-off heatmap (coordination / SoD surface)."""
    fig, ax = plt.subplots(figsize=(0.45 * len(H) + 3, 0.45 * len(H) + 2))
    im = ax.imshow(H.values, cmap="magma")
    ax.set_xticks(range(len(H.columns)), H.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(H.index)), H.index, fontsize=6)
    ax.set_xlabel("to resource"); ax.set_ylabel("from resource")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="hand-off count")
    fig.tight_layout()
    return _save(fig, name)


def fig_value_phase(profile_df, title, name="fig_value_phase"):
    """Penalty-mass share by value band (financial materiality of deviation)."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bands = profile_df["value_band"].astype(str)
    ax.bar(bands, profile_df["penalty_mass_share"], color="#9467bd", alpha=0.85)
    ax.set_ylabel("share of penalty mass")
    ax.set_xlabel("exposure (value) band — Q1 lowest … Q5 highest")
    ax2 = ax.twinx()
    ax2.plot(bands, profile_df["mean_penalty"], color="black", marker="o", ms=4)
    ax2.set_ylabel("mean penalty")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, name)
