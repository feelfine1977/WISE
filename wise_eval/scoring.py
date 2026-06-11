"""
wise.scoring — Phase 2: score cases, keep layer/constraint evidence
(Fig. 2, blue band; paper Sec. IV-D).

Applicability-aware case score under view p:

    S^(p)(σ) = 1 −  Σ_{c∈C_app(σ)} w_c^(p) ν_c(σ)
                    ───────────────────────────────
                    Σ_{c∈C_app(σ)} w_c^(p)

so S = 0.8 means "the applicable constraints have a weighted average
violation of 0.2 under view p". Cases with no positively weighted
applicable constraint are unscored (excluded from aggregation).

Layer contributions (the drill-down currency of WISE):

    1 − S^(p)(σ) = Σ_λ Δ_λ^(p)(σ),   Δ_λ^(p)(σ) = Σ_{c∈C_λ} w̃_c,σ^(p) ν_c(σ)

with case-specific *effective* weights w̃ (raw weights renormalised over
the applicable set). Because layers partition the norm, the case penalty
decomposes *exactly* into layer terms — nothing is lost or double
counted, which is what makes slice explanations trustworthy.

Implementation note: the BPIC'19 instantiation uses the "layer-balanced"
mode — within-layer weighted penalty per layer, then a view-weighted
average across applicable layers. This is the two-stage elicitation
applied at scoring time; with full applicability it coincides with the
flat formula above.
"""
import numpy as np
import pandas as pd


def compute_layer_penalties(df_violations: pd.DataFrame, norm: dict) -> pd.DataFrame:
    """Within-layer weighted mean violation per case and layer.

    For layer λ:  pen_λ(σ) = Σ_{c∈C_λ∩C_app(σ)} b̂_c ν_c(σ) / Σ b̂_c,
    NaN when no constraint of λ is applicable to σ (layer not scored).
    """
    out = pd.DataFrame({"case_id": df_violations["case_id"]})
    for layer_id in norm["layers"]:
        lcs = [c for c in norm["constraints"] if c["layer_id"] == layer_id]
        w = np.array([c.get("within_layer_weight", 1.0) for c in lcs], dtype=float)
        w = w / w.sum()
        cols = [c["id"] for c in lcs]
        mat = df_violations[cols]
        num = mat.fillna(0).mul(w, axis=1).sum(axis=1)
        den = mat.notna().astype(float).mul(w, axis=1).sum(axis=1)
        out[layer_id] = num.div(den).where(den > 0, np.nan)
    return out


def compute_view_scores(df_layer_penalties: pd.DataFrame, norm: dict):
    """View-specific case scores S^(p)(σ) and layer contributions Δ_λ^(p)(σ).

    Returns (df_view_scores, df_view_layer_contribs). Contributions are
    constructed so that  Σ_λ contrib__p__λ(σ) = 1 − S^(p)(σ)  exactly.
    """
    df_view_scores = pd.DataFrame({"case_id": df_layer_penalties["case_id"]})
    df_contribs = pd.DataFrame({"case_id": df_layer_penalties["case_id"]})
    for view_name, view_info in norm["views"].items():
        lw = pd.Series(view_info["layer_weights"])
        block = df_layer_penalties[list(lw.index)]
        num = block.fillna(0).mul(lw, axis=1).sum(axis=1)
        den = block.notna().astype(float).mul(lw, axis=1).sum(axis=1)
        penalty = num.div(den).where(den > 0, np.nan)
        df_view_scores[f"score__{view_name}"] = 1 - penalty
        for layer_id in lw.index:
            df_contribs[f"contrib__{view_name}__{layer_id}"] = (
                block[layer_id].fillna(0) * lw[layer_id]
            ).div(den).where(den > 0, 0)
    return df_view_scores, df_contribs


def assemble_case_scores(case_df, df_violations, df_layer_penalties,
                         df_view_scores, df_contribs) -> pd.DataFrame:
    """One wide frame: features + ν_c + layer penalties + S^(p) + Δ_λ^(p)."""
    return (case_df
            .merge(df_violations, on="case_id")
            .merge(df_layer_penalties, on="case_id")
            .merge(df_view_scores, on="case_id")
            .merge(df_contribs, on="case_id"))
