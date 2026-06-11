"""
WISE — Weighted Insights for Evaluating Efficiency
===================================================

A *basic*, paper-faithful implementation of the WISE method
(Jessen, Fahland, Zerbato — "WISE: Actionable Norm-Based Scoring for
Process Mining") for the BPIC'19 purchase-to-pay benchmark.

The package mirrors the workflow of Figure 2 of the paper:

    Before computation  ──  governance framing
        wise.data           load + prepare the event log, fix case notion
        wise.features       per-case feature extraction (slice keys included)

    Phase 1 ──  instantiate norm, layers, and views
        wise.norm           versioned norm N = (C, Λ, lay), views p,
                            two-stage weight elicitation  w_c^(p) = a_λ^(p) · b̂_c

    Phase 2 ──  score cases, keep layer/constraint evidence
        wise.constraints    bounded violation signals ν_c(σ) ∈ [0,1]
        wise.scoring        case scores S^(p)(σ), layer contributions Δλ^(p)(σ)

    Phase 3 ──  prioritise documents and ownership slices
        wise.prioritization PI = scale × underperformance, shrinkage-stable PI

    Governance loop ──  review → validate → refine → rerun
        wise.governance     right-censoring, event replication,
                            censoring-robust gap retention (paper Table X)

    wise.viz                all evaluation figures of Section V
"""
from . import config, constraints, data, features, governance, norm
from . import prioritization, scoring, viz

__version__ = "1.0-basic"
__all__ = [
    "config", "data", "features", "norm", "constraints",
    "scoring", "prioritization", "governance", "viz",
]
