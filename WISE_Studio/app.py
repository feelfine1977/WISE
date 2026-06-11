"""
WISE Studio — guided, norm-based event-log analysis for ANY event log.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Three steps:
  1. Upload & Map   — load a CSV (or demo data); WISE auto-detects which column
                      is the case id / activity / timestamp / resource and which
                      are business dimensions; you confirm or override.
  2. Build the Norm — a guided builder: data-grounded suggestions, add/edit the
                      five constraint types with activity pickers and sliders,
                      group into layers, weight stakeholder views.
  3. Analyze        — one click: baselines, ranked slice backlog with hotspot
                      types, risk matrix, layer deltas, ownership heatmap,
                      leading constraints, downloads.

All heavy lifting lives in the headless-tested `wise` package; this file is a
thin UI layer.
"""
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from wise import dataio, features, norm as wnorm, balance_features, constraints
from wise import scoring, prioritize, viz
from wise.dataio import load_mapped, load_sample, mapped_columns
from wise.mapping import (detect_mapping, empty_mapping, profile_columns,
                          validate_mapping, slug)
from wise.demo import generate_demo_log

st.set_page_config(page_title="WISE Studio", page_icon="🧭", layout="wide")

LEVEL_ICON = {"ok": "✅", "warn": "⚠️", "error": "❌"}
CTYPES = ["presence", "lag", "singularity", "exclusion", "balance"]
CTYPE_HELP = {
    "presence":    "A required activity must occur (e.g. every case must reach 'Payment').",
    "lag":         "After activity A, activity B should follow within N days (timeliness).",
    "singularity": "An activity should not repeat too often (rework / fragmentation).",
    "exclusion":   "An activity should not occur at all (cancellations, corrections).",
    "balance":     "Two numeric totals should match within a tolerance (value consistency).",
}

# ---------------------------------------------------------------- state
def _init_state():
    ss = st.session_state
    ss.setdefault("df_raw", None)          # profiling frame (may be a sample)
    ss.setdefault("source_name", None)
    ss.setdefault("source", None)          # ("path", p) | ("buffer", bytes)
    ss.setdefault("is_sample", False)
    ss.setdefault("n_loaded_events", None)
    ss.setdefault("mapping", empty_mapping())
    ss.setdefault("confidence", {})
    ss.setdefault("events", None)
    ss.setdefault("fc", None)
    ss.setdefault("norm", {"name": "my norm", "version": "0.1",
                           "layers": dict(wnorm.DEFAULT_LAYERS),
                           "constraints": [], "views": {}})
    ss.setdefault("result", None)

_init_state()
ss = st.session_state


SAMPLE_ROWS = 150_000  # profiling sample size for large files


def _rebuild_canonical():
    """Full mapped load (only the mapped columns) + case features.
    Frees the raw profiling frame afterwards — on large logs the canonical
    events frame (categorical dtypes) is what we keep, nothing else."""
    import io as _io
    kind, src = ss.source
    if kind == "path":
        ss.events, _ = load_mapped(src, ss.mapping)
    else:
        ss.events, _ = load_mapped(_io.BytesIO(src), ss.mapping)
    ss.fc = features.build_case_features(ss.events, ss.mapping)
    ss.n_loaded_events = len(ss.events)
    ss.df_raw = None          # release the profiling frame
    ss.result = None


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧭 WISE Studio")
    st.caption("Norm-based scoring & slice-first prioritisation for any event log.")
    step = st.radio("Steps", ["1 · Upload & Map", "2 · Build the Norm", "3 · Analyze"],
                    label_visibility="collapsed")
    st.divider()
    if ss.source_name is not None:
        if ss.n_loaded_events:
            st.success(f"Data: **{ss.source_name}**  \n{ss.n_loaded_events:,} events loaded")
        elif ss.df_raw is not None:
            note = " (profiling sample)" if ss.is_sample else ""
            st.success(f"Data: **{ss.source_name}**  \n{len(ss.df_raw):,} rows{note}")
    if ss.events is not None:
        st.info(f"Mapped: {ss.events['case_id'].nunique():,} cases, "
                f"{ss.events['activity'].nunique()} activities")
    n_c = len(ss.norm["constraints"])
    if n_c:
        st.info(f"Norm: {n_c} constraints, {len(ss.norm['views'])} view(s)")

# ============================================================== STEP 1
if step.startswith("1"):
    st.header("Step 1 — Upload your event log & map the fields")
    st.markdown("Load any CSV event log. WISE profiles the columns, **guesses** which one is the "
                "case id, activity, timestamp, resource and which are business **dimensions** — "
                "you confirm or override every guess.")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        up = st.file_uploader("Upload a CSV", type=["csv"])
    with c2:
        path_in = st.text_input("…or a file path on this machine",
                                placeholder="C:\\data\\my_event_log.csv")
    with c3:
        st.write("")
        demo_btn = st.button("🎲 Use demo data", use_container_width=True)

    loaded = False
    if demo_btn:
        ss.df_raw = generate_demo_log(3000)
        ss.source_name = "demo order-to-delivery log"
        ss.source = ("buffer", ss.df_raw.to_csv(index=False).encode())
        ss.is_sample = False
        loaded = True
    elif up is not None and (ss.source_name != up.name):
        raw = up.getvalue()
        big = len(raw) > 120 * 1024 * 1024          # >120 MB → profile a sample
        df, enc = load_sample(io.BytesIO(raw), sample_rows=SAMPLE_ROWS) if big \
            else dataio.read_csv_any(io.BytesIO(raw))
        ss.df_raw, ss.source_name = df, up.name
        ss.source, ss.is_sample = ("buffer", raw), big
        ss.mapping["encoding"] = enc
        loaded = True
    elif path_in and Path(path_in).exists() and ss.source_name != path_in:
        big = Path(path_in).stat().st_size > 120 * 1024 * 1024
        df, enc = load_sample(path_in, sample_rows=SAMPLE_ROWS) if big \
            else dataio.read_csv_any(path_in)
        ss.df_raw, ss.source_name = df, path_in
        ss.source, ss.is_sample = ("path", path_in), big
        ss.mapping["encoding"] = enc
        loaded = True

    if loaded:
        m, conf, _ = detect_mapping(ss.df_raw)
        ss.mapping.update(m)
        ss.confidence = conf
        ss.events = ss.fc = ss.result = None
        st.toast("Columns profiled — review the suggested mapping below.")

    if ss.df_raw is None and ss.events is None:
        st.info("Upload a CSV, enter a path, or click **Use demo data** to try the app instantly.")
        st.stop()
    if ss.df_raw is None and ss.events is not None:
        st.success(f"Mapping applied — {ss.fc['case_id'].nunique():,} cases / "
                   f"{ss.n_loaded_events:,} events loaded (only the mapped columns). "
                   "Load a different file above, or continue with **Step 2**.")
        st.stop()
    if ss.is_sample:
        st.warning(f"Large file detected — profiling and mapping run on the first "
                   f"{len(ss.df_raw):,} rows. The **full file** is parsed (mapped "
                   f"columns only) when you click *Apply mapping*. For files of "
                   f"several hundred MB, prefer the file-path input over the "
                   f"browser upload.")

    st.subheader("Column profile")
    st.dataframe(profile_columns(ss.df_raw).round(3), use_container_width=True, height=240)

    st.subheader("Field mapping")
    st.caption("Suggestions are marked with a confidence badge — override anything via the dropdowns.")
    cols = ["— none —"] + list(ss.df_raw.columns)

    def pick(role, label, help_txt):
        cur = ss.mapping.get(role)
        badge = f"  ·  guess {ss.confidence.get(role, 0):.0%}" if ss.confidence.get(role) else ""
        idx = cols.index(cur) if cur in cols else 0
        sel = st.selectbox(label + badge, cols, index=idx, help=help_txt, key=f"map_{role}")
        ss.mapping[role] = None if sel == "— none —" else sel

    a, b = st.columns(2)
    with a:
        pick("case_id", "🔑 Case ID (required)", "One value per process instance (order, ticket, PO item…).")
        pick("activity", "🏃 Activity (required)", "What happened — the event/step name.")
        pick("timestamp", "🕒 Timestamp (required)", "When it happened. Must parse as a date/time.")
    with b:
        pick("resource", "👤 Resource (optional)", "Who/what performed the event.")
        pick("document_id", "📄 Roll-up ID (optional)", "A parent unit (e.g. PO header) for governance roll-ups.")

    dims_default = [d for d in ss.mapping.get("dimensions", []) if d in ss.df_raw.columns]
    ss.mapping["dimensions"] = st.multiselect(
        "🧭 Business dimensions (slicing attributes)", list(ss.df_raw.columns), default=dims_default,
        help="Case-constant attributes you steer by: company, region, vendor, product type…")
    nums_default = [d for d in ss.mapping.get("numeric_attributes", []) if d in ss.df_raw.columns]
    ss.mapping["numeric_attributes"] = st.multiselect(
        "💶 Numeric attributes (optional)", list(ss.df_raw.columns), default=nums_default,
        help="Event-level amounts/quantities — enables value (balance) checks.")

    issues = validate_mapping(ss.df_raw, ss.mapping)
    st.subheader("Mapping check")
    for level, msg in issues:
        st.markdown(f"{LEVEL_ICON[level]} {msg}")

    ok = not any(l == "error" for l, _ in issues)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Apply mapping & build case table", type="primary",
                     disabled=not ok, use_container_width=True):
            with st.spinner("Loading the full file (mapped columns only) and "
                            "building the case table — large files can take a minute…"):
                _rebuild_canonical()
            st.success(f"Done — {ss.fc['case_id'].nunique():,} cases ready. "
                       "Continue with **Step 2 — Build the Norm**.")
    with c2:
        st.download_button("💾 Download log_mapping.json",
                           json.dumps(ss.mapping, indent=2),
                           "log_mapping.json", "application/json",
                           use_container_width=True)

# ============================================================== STEP 2
elif step.startswith("2"):
    st.header("Step 2 — Build the norm (guided)")
    if ss.fc is None:
        st.warning("Finish **Step 1** first (apply the mapping)."); st.stop()

    norm = ss.norm
    tab_sugg, tab_edit, tab_views = st.tabs(
        ["✨ Suggestions from your data", "🧱 Constraints & layers", "🎚️ Stakeholder views"])

    # ---- suggestions ----
    with tab_sugg:
        st.markdown("The norm says **what good execution looks like**. Start from "
                    "data-grounded suggestions, then refine — every threshold below "
                    "comes from *your* log's observed behaviour.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Activity frequencies** *(what happens, how often, how repetitive)*")
            st.dataframe(wnorm.activity_stats(ss.events).round(3),
                         use_container_width=True, height=260)
        with c2:
            st.markdown("**Common handovers** *(directly-follows pairs and their lags)*")
            st.dataframe(wnorm.common_pairs(ss.events, top=10).round(2),
                         use_container_width=True, height=260)
        if st.button("✨ Suggest a starter norm from this log", type="primary"):
            sugg = wnorm.suggest_norm(ss.events, ss.fc)
            norm["layers"] = sugg["layers"]
            norm["constraints"] = sugg["constraints"]
            if not norm["views"]:
                norm["views"] = sugg["views"]
            ss.result = None
            st.success(f"Added {len(sugg['constraints'])} suggested constraints — "
                       "review them in **Constraints & layers**.")

    # ---- constraints & layers ----
    with tab_edit:
        acts = sorted(ss.events["activity"].unique())
        num_attrs = [a for a in ss.mapping.get("numeric_attributes", [])
                     if a in ss.events.columns]
        dim_cols = [d for d in ss.mapping.get("dimensions", []) if d in ss.fc.columns]

        st.markdown(f"**Current constraints ({len(norm['constraints'])})**")
        if norm["constraints"]:
            view_df = pd.DataFrame([
                {"id": c["id"], "layer": c["layer"], "type": c["type"],
                 "what": c.get("description", ""),
                 "applies to": "; ".join(f"{f['column']}∈{f['values']}"
                                          for f in c.get("applicability", []) or []) or "all cases"}
                for c in norm["constraints"]])
            st.dataframe(view_df, use_container_width=True, height=220)
            del_id = st.selectbox("Delete a constraint",
                                  ["—"] + [c["id"] for c in norm["constraints"]])
            if del_id != "—" and st.button("🗑️ Delete selected"):
                norm["constraints"] = [c for c in norm["constraints"] if c["id"] != del_id]
                ss.result = None
                st.rerun()
        else:
            st.info("No constraints yet — use ✨ Suggestions or add one below.")

        st.divider()
        st.markdown("**➕ Add a constraint**")
        ctype = st.selectbox("Type", CTYPES, format_func=lambda t: f"{t} — {CTYPE_HELP[t]}")
        layer = st.selectbox("Layer (business-facing group)",
                             list(norm["layers"].keys()),
                             format_func=lambda l: f"{l} · {norm['layers'][l]['name']}")
        new_c = {"layer": layer, "type": ctype}

        if ctype == "presence":
            new_c["activity"] = st.selectbox("Required activity", acts)
            new_c["min_count"] = st.number_input("Minimum occurrences", 1, 20, 1)
            new_c["id"] = f"c_presence_{slug(new_c['activity'])}"
        elif ctype == "lag":
            c1, c2 = st.columns(2)
            new_c["from_activity"] = c1.selectbox("From activity (A)", acts)
            new_c["to_activity"] = c2.selectbox("To activity (B)", acts, index=min(1, len(acts)-1))
            c1, c2 = st.columns(2)
            new_c["threshold_days"] = c1.slider("No penalty up to (days)", 0.0, 90.0, 10.0, 0.5)
            new_c["saturation_days"] = c2.slider("Max penalty after +(days)", 0.5, 180.0, 20.0, 0.5)
            new_c["id"] = f"c_lag_{slug(new_c['from_activity'])}_to_{slug(new_c['to_activity'])}"
        elif ctype == "singularity":
            new_c["activity"] = st.selectbox("Repeating activity", acts)
            c1, c2 = st.columns(2)
            new_c["allowed"] = c1.slider("Allowed repetitions (no penalty)", 0, 20, 2)
            new_c["saturation"] = c2.slider("Saturation width", 1, 20, 3)
            new_c["id"] = f"c_repeat_{slug(new_c['activity'])}"
        elif ctype == "exclusion":
            new_c["activity"] = st.selectbox("Forbidden / exception activity", acts)
            new_c["id"] = f"c_excl_{slug(new_c['activity'])}"
        else:  # balance
            if not num_attrs:
                st.warning("Balance checks need a numeric attribute — add one in Step 1.")
            new_c["attribute"] = st.selectbox("Numeric attribute", num_attrs or ["—"])
            c1, c2 = st.columns(2)
            new_c["activities_x"] = c1.multiselect("Total over activities (X)", acts)
            new_c["activities_y"] = c2.multiselect("…should match total over (Y)", acts)
            c1, c2 = st.columns(2)
            new_c["tolerance"] = c1.slider("Tolerated relative mismatch", 0.0, 0.5, 0.05, 0.01)
            new_c["saturation"] = c2.slider("Saturation width", 0.05, 1.0, 0.20, 0.05)
            new_c["id"] = f"c_balance_{slug(new_c.get('attribute', 'attr'))}"

        with st.expander("Optional: restrict where this check applies (applicability)"):
            st.caption("Evaluate this constraint only for cases matching a dimension filter "
                       "— e.g. only one flow type or one region. Other cases are *not applicable* "
                       "(excluded, not penalised).")
            app_col = st.selectbox("Dimension", ["—"] + dim_cols)
            if app_col != "—":
                vals = st.multiselect("Applies when value is one of",
                                      sorted(ss.fc[app_col].astype(str).unique()))
                if vals:
                    new_c["applicability"] = [{"column": app_col, "values": vals}]

        new_c["description"] = st.text_input("Description (business language)",
                                             value=CTYPE_HELP[ctype])
        if st.button("➕ Add constraint", type="primary"):
            if any(c["id"] == new_c["id"] for c in norm["constraints"]):
                st.error(f"A constraint with id '{new_c['id']}' already exists.")
            else:
                norm["constraints"].append(new_c)
                ss.result = None
                st.success(f"Added {new_c['id']}")
                st.rerun()

        st.divider()
        with st.expander("Manage layers"):
            for lid in list(norm["layers"]):
                c1, c2 = st.columns([3, 1])
                norm["layers"][lid]["name"] = c1.text_input(
                    f"{lid} name", norm["layers"][lid]["name"], key=f"lay_{lid}")
                used = any(c["layer"] == lid for c in norm["constraints"])
                if c2.button(f"Remove {lid}", disabled=used, key=f"rm_{lid}",
                             help="Layers in use cannot be removed."):
                    del norm["layers"][lid]
                    st.rerun()
            new_lid = st.text_input("New layer id (e.g. L6)")
            new_lname = st.text_input("New layer name")
            if st.button("Add layer") and new_lid and new_lname and new_lid not in norm["layers"]:
                norm["layers"][new_lid] = {"name": new_lname}
                st.rerun()

    # ---- views ----
    with tab_views:
        st.markdown("A **view** weights the layers from one stakeholder's perspective — "
                    "same checks, different priorities. Weights are renormalised to sum to 1.")
        vname = st.text_input("Add a view (e.g. Finance, Operations, Compliance)")
        if st.button("➕ Add view") and vname and vname not in norm["views"]:
            n = len(norm["layers"])
            norm["views"][vname] = {"layer_weights": {l: round(1/n, 3) for l in norm["layers"]}}
            st.rerun()
        for v in list(norm["views"]):
            with st.expander(f"View: {v}", expanded=True):
                w = norm["views"][v]["layer_weights"]
                for l in norm["layers"]:
                    w[l] = st.slider(f"{l} · {norm['layers'][l]['name']}",
                                     0.0, 1.0, float(w.get(l, 0.0)), 0.01, key=f"w_{v}_{l}")
                tot = sum(w.values()) or 1.0
                st.caption(f"Sum = {tot:.2f} → normalised at scoring time.")
                if st.button(f"Remove view '{v}'", key=f"rmv_{v}"):
                    del norm["views"][v]
                    st.rerun()

    st.divider()
    st.subheader("Norm check")
    for level, msg in wnorm.validate_norm(norm, ss.fc):
        st.markdown(f"{LEVEL_ICON[level]} {msg}")
    st.download_button("💾 Download norm.json", json.dumps(norm, indent=2),
                       "norm.json", "application/json")
    upn = st.file_uploader("…or load an existing norm.json", type=["json"], key="norm_up")
    if upn is not None:
        ss.norm = json.load(upn)
        ss.result = None
        st.success("Norm loaded.")
        st.rerun()

# ============================================================== STEP 3
else:
    st.header("Step 3 — Analyze")
    if ss.fc is None:
        st.warning("Finish **Step 1** first."); st.stop()
    if not ss.norm["constraints"]:
        st.warning("Build a norm in **Step 2** first (or click ✨ Suggest)."); st.stop()

    dim_cols = [d for d in ss.mapping.get("dimensions", []) if d in ss.fc.columns]
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        slice_dims = st.multiselect("Slice by (ownership dimensions)", dim_cols,
                                    default=dim_cols[:2],
                                    help="Cases are grouped into slices by these attributes; "
                                         "the backlog ranks the slices.")
    with c2:
        gamma = st.slider("Stability γ (shrinkage)", 0, 100, 20,
                          help="Higher = small slices pulled harder to the global mean "
                               "(protects against noisy tiny slices).")
    with c3:
        st.write("")
        go = st.button("🚀 Run analysis", type="primary", use_container_width=True,
                       disabled=not slice_dims)

    if go:
        with st.spinner("Scoring cases against the norm and ranking slices…"):
            nrm = ss.norm
            fc = balance_features.add_balance_totals(ss.fc, ss.events, nrm)
            viol = constraints.build_violation_matrix(fc, nrm)
            scores = scoring.score_cases(fc, viol, nrm)
            res = {"scores": scores, "slice_dims": slice_dims,
                   "views": list(nrm["views"].keys()),
                   "backlogs": {}, "typed": {}}
            for v in res["views"]:
                bl = prioritize.slice_backlog(scores, v, slice_dims, gamma=gamma)
                res["backlogs"][v] = bl
                res["typed"][v] = prioritize.classify_hotspots(bl)
            ss.result = res

    res = ss.result
    if res is None:
        st.info("Pick slicing dimensions and click **Run analysis**.")
        st.stop()

    scores, nrm = res["scores"], ss.norm
    st.subheader("Baselines")
    bc = st.columns(len(res["views"]))
    for col, v in zip(bc, res["views"]):
        mu = scores[f"score__{v}"].mean()
        col.metric(f"{v} baseline μ̄", f"{mu:.3f}", f"avg violation {1-mu:.1%}",
                   delta_color="off")

    view = st.selectbox("View for the drill-down", res["views"])
    typed = res["typed"][view]

    t1, t2, t3, t4 = st.tabs(["📋 Backlog", "🗺️ Risk matrix & ownership",
                              "🧩 Why? (layers & constraints)", "⬇️ Downloads"])
    with t1:
        st.markdown("**Ranked slice backlog** — `reservoir` = big & mildly bad (manage the "
                    "queue), `severity` = small & acutely bad (fix the local cause), "
                    "`mechanism` = a specific recurring defect (redesign that step).")
        st.dataframe(typed.round(4), use_container_width=True)
        st.pyplot(viz.fig_backlog_pareto(res["backlogs"][view], view))
        st.pyplot(viz.fig_score_distribution(scores, view))
    with t2:
        if len(typed):
            st.pyplot(viz.fig_risk_matrix(typed, res["slice_dims"], view))
        if len(res["slice_dims"]) >= 2:
            fig = viz.fig_ownership_heatmap(res["backlogs"][view], res["slice_dims"], view)
            if fig:
                st.pyplot(fig)
    with t3:
        if len(typed):
            options = [tuple(r[c] for c in res["slice_dims"]) for _, r in typed.iterrows()]
            sel = st.selectbox("Drill into slice",
                               options, format_func=lambda t: " × ".join(map(str, t)))
            deltas = prioritize.layer_deltas(scores, view, res["slice_dims"], nrm)
            key = sel if len(res["slice_dims"]) > 1 else sel[0]
            try:
                st.pyplot(viz.fig_layer_deltas(deltas.loc[[key]], nrm, view))
            except KeyError:
                st.info("No delta available for this slice.")
            mask = pd.Series(True, index=scores.index)
            for c, val in zip(res["slice_dims"], sel):
                mask &= scores[c].astype(str) == str(val)
            st.markdown("**Leading constraints in this slice** *(which checks carry the penalty)*")
            st.dataframe(prioritize.leading_constraints(scores, mask, nrm).round(3),
                         use_container_width=True)
    with t4:
        st.download_button("backlog.csv",
                           res["backlogs"][view].to_csv(index=False),
                           f"backlog_{view.lower()}.csv", "text/csv")
        if st.checkbox("Prepare full case_scores.csv export "
                       "(can be large — one row per case, all columns)"):
            st.download_button("case_scores.csv (full)",
                               scores.to_csv(index=False),
                               "case_scores.csv", "text/csv")
        st.download_button("norm.json", json.dumps(nrm, indent=2),
                           "norm.json", "application/json")
        st.download_button("log_mapping.json", json.dumps(ss.mapping, indent=2),
                           "log_mapping.json", "application/json")
        st.caption("Tip: the same analysis runs headless — "
                   "`python -m wise.pipeline data.csv log_mapping.json norm.json "
                   "--dims <dim1> <dim2> --out output`")
