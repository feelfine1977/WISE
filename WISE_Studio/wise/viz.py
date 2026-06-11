"""Generic figures. Each returns a matplotlib Figure (Streamlit-friendly) and
optionally saves a PNG when out_dir is given."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, out_dir, name):
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig


def fig_score_distribution(scores, view, out_dir=None):
    s = scores[f"score__{view}"].dropna()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(s, bins=40, color="#1f77b4", alpha=0.8)
    ax.axvline(s.mean(), color="black", ls="--", lw=1.2,
               label=f"baseline μ̄ = {s.mean():.3f}")
    ax.set_xlabel("case score (1 = fully meets the norm)")
    ax.set_ylabel("cases")
    ax.set_title(f"Case-score distribution — {view} view")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, f"score_distribution_{view.lower()}")


def fig_backlog_pareto(backlog, view, threshold=0.80, out_dir=None):
    pi = backlog["PI_stable"].to_numpy()
    cum = np.cumsum(pi) / max(pi.sum(), 1e-12)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(range(1, len(cum) + 1), cum, lw=1.8, color="#9467bd")
    ax.axhline(threshold, color="grey", ls="--", lw=1)
    k = int(np.searchsorted(cum, threshold) + 1) if len(cum) else 0
    ax.set_title(f"Priority concentration — top {k} slices capture "
                 f"{threshold:.0%} of priority mass ({view})")
    ax.set_xlabel("top-k slices, ranked by stable PI")
    ax.set_ylabel("cumulative share of stable PI")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    return _save(fig, out_dir, f"backlog_pareto_{view.lower()}")


def fig_risk_matrix(typed, group_cols, view, out_dir=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    markers = {"reservoir": "s", "mechanism": "o", "severity": "^"}
    vmax = max(typed["PI_stable"].max(), 1e-9)
    sc = None
    for typ, sub in typed.groupby("hotspot_type"):
        sc = ax.scatter(sub["cases"], sub["stable_gap"],
                        s=50 + 220*sub["PI_stable"]/vmax, c=sub["PI_stable"],
                        cmap="YlOrRd", vmin=0, vmax=vmax,
                        marker=markers.get(typ, "o"), edgecolor="k", lw=0.6, label=typ)
    for _, r in typed.head(6).iterrows():
        ax.annotate(" × ".join(str(r[c]) for c in group_cols),
                    (r["cases"], r["stable_gap"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("slice volume (cases, log scale)")
    ax.set_ylabel("stabilised underperformance gap")
    ax.set_title(f"Risk matrix — {view}: right = manage queue, up = fix local cause")
    ax.legend(title="hotspot type", fontsize=8)
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="stable PI")
    fig.tight_layout()
    return _save(fig, out_dir, f"risk_matrix_{view.lower()}")


def fig_layer_deltas(deltas, norm, view, out_dir=None):
    names = [norm["layers"][l]["name"] for l in deltas.columns]
    n = max(len(deltas), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.4*n, 3.4), sharey=True)
    for ax, (idx, row) in zip(np.atleast_1d(axes), deltas.iterrows()):
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in row.values]
        ax.barh(range(len(row)), row.values, color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(row)), names, fontsize=7)
        ax.set_title(idx if isinstance(idx, str) else " × ".join(map(str, idx)), fontsize=9)
        ax.set_xlabel("Δ vs global")
    fig.suptitle(f"Layer-delta profiles ({view}) — red = mechanism stronger in slice", y=1.04)
    fig.tight_layout()
    return _save(fig, out_dir, f"layer_deltas_{view.lower()}")


def fig_ownership_heatmap(backlog, group_cols, view, out_dir=None):
    if len(group_cols) < 2:
        return None
    pivot = backlog.pivot_table(index=group_cols[0], columns=group_cols[1],
                                values="PI_stable", aggfunc="sum", fill_value=0.0)
    rel = pivot - pivot.values.mean()
    fig, ax = plt.subplots(figsize=(max(7, 0.5*rel.shape[1]+3), max(3.2, 0.4*rel.shape[0]+1.6)))
    vmax = np.abs(rel.values).max() or 1.0
    im = ax.imshow(rel.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(rel.shape[1]), rel.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(rel.shape[0]), rel.index, fontsize=8)
    ax.set_title(f"Priority mass by {group_cols[0]} × {group_cols[1]} ({view}) — red = above average")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _save(fig, out_dir, f"ownership_heatmap_{view.lower()}")
