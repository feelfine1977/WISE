"""One-call pipeline: data → mapping → features → norm → scores → backlog → figures.

Programmatic:
    from wise.pipeline import run
    result = run("events.csv", "log_mapping.json", "norm.json",
                 slice_dims=["region", "product_type"], out_dir="output")

CLI:
    python -m wise.pipeline events.csv log_mapping.json norm.json \
        --dims region product_type --out output
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import dataio, features, norm as wnorm, constraints, balance_features
from . import scoring, prioritize, viz
from .mapping import load_mapping, validate_mapping


def run(csv_path, mapping_path, norm_path, slice_dims=None, gamma=20.0,
        out_dir="output", views=None, top_n=12, make_figures=True):
    mapping = load_mapping(mapping_path) if isinstance(mapping_path, (str, Path)) else mapping_path
    nrm = wnorm.load_norm(norm_path) if isinstance(norm_path, (str, Path)) else norm_path

    df_raw, enc = dataio.read_csv_any(csv_path, encoding=mapping.get("encoding", "utf-8"))
    issues = validate_mapping(df_raw, mapping)
    if any(l == "error" for l, _ in issues):
        raise ValueError("Mapping errors: " + "; ".join(m for l, m in issues if l == "error"))

    events = dataio.canonicalize(df_raw, mapping)
    fc = features.build_case_features(events, mapping)
    fc = balance_features.add_balance_totals(fc, events, nrm)
    viol = constraints.build_violation_matrix(fc, nrm)
    scores = scoring.score_cases(fc, viol, nrm)

    slice_dims = slice_dims or mapping.get("dimensions", [])[:2]
    views = views or list(nrm["views"].keys())
    out = {"events": events, "case_features": fc, "scores": scores,
           "mapping_issues": issues, "norm_issues": wnorm.validate_norm(nrm, fc),
           "baselines": {v: float(scores[f"score__{v}"].mean()) for v in views},
           "backlogs": {}, "typed": {}, "concentration": {}, "figures_dir": None}

    for v in views:
        bl = prioritize.slice_backlog(scores, v, slice_dims, gamma=gamma)
        out["backlogs"][v] = bl
        out["typed"][v] = prioritize.classify_hotspots(bl, top_n=top_n)
        out["concentration"][v], _ = prioritize.pareto_concentration(bl)

    if make_figures and out_dir:
        fig_dir = Path(out_dir) / "figures"
        out["figures_dir"] = str(fig_dir)
        for v in views:
            viz.fig_score_distribution(scores, v, fig_dir)
            viz.fig_backlog_pareto(out["backlogs"][v], v, out_dir=fig_dir)
            if len(out["typed"][v]):
                viz.fig_risk_matrix(out["typed"][v], slice_dims, v, fig_dir)
                top3 = [tuple(r[c] for c in slice_dims)
                        for _, r in out["typed"][v].head(3).iterrows()]
                deltas = prioritize.layer_deltas(scores, v, slice_dims, nrm)
                # tolerate single-dim slices (index not MultiIndex)
                try:
                    sel = deltas.loc[top3 if len(slice_dims) > 1 else [t[0] for t in top3]]
                    viz.fig_layer_deltas(sel, nrm, v, fig_dir)
                except KeyError:
                    pass
            viz.fig_ownership_heatmap(out["backlogs"][v], slice_dims, v, fig_dir)
        # tables
        tdir = Path(out_dir) / "tables"; tdir.mkdir(parents=True, exist_ok=True)
        for v in views:
            out["backlogs"][v].to_csv(tdir / f"backlog_{v.lower()}.csv", index=False)
        scores.to_csv(tdir / "case_scores.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description="Run the WISE pipeline on any event log.")
    ap.add_argument("csv"); ap.add_argument("mapping"); ap.add_argument("norm")
    ap.add_argument("--dims", nargs="*", default=None)
    ap.add_argument("--gamma", type=float, default=20.0)
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    res = run(a.csv, a.mapping, a.norm, slice_dims=a.dims, gamma=a.gamma, out_dir=a.out)
    print("Baselines:", {k: round(v, 3) for k, v in res["baselines"].items()})
    for v, bl in res["backlogs"].items():
        print(f"\nTop slices — {v}:")
        print(bl.head(8).to_string(index=False))
    print(f"\nFigures: {res['figures_dir']}  |  Tables: {Path(a.out)/'tables'}")


if __name__ == "__main__":
    main()
