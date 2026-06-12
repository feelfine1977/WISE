# READ_WISE — Understanding and Using the WISE Framework

> A complete walkthrough of **WISE (Weighted Insights for Evaluating Efficiency)** — what it is for,
> how every step works, and how to apply it to a new dataset.
>
> Built from *WISE: Actionable Norm-Based Scoring for Process Mining* (the paper) and the
> **EXTENDED / annotated** BPIC'19 analysis ([WISE_BPIC19_full_analysis_EXTENDED.ipynb](WISE_BPIC19_full_analysis_EXTENDED.ipynb)).

---

## 1. The goal of the framework

Most process-mining dashboards tell you **what happened** (variants, frequencies, durations). They
rarely tell you **where performance is weakest, why it is weak, and where to spend scarce
improvement effort first** — and they often manufacture false alarms by treating every deviation as
a problem regardless of context.

WISE is a **norm-based, multi-stakeholder prioritisation framework** that answers three questions in
one auditable pipeline:

1. **Where** does deviation concentrate? → ranked *slices* (e.g. company × spend area) and documents.
2. **For whom** does it matter? → role-based *views* (Finance, Logistics, Compliance, Automation).
3. **Why** does a hotspot underperform? → an *exact* breakdown into business-facing *layers*, drilled
   down to the individual constraint, variant, vendor, and resource hand-off.

The design principles that make it different:

- **Explicit norm, not learned "normal".** Expected behaviour is declared as machine-checkable
  constraints, so a deviation is measured against what *should* happen, not against the statistical
  majority.
- **Applicability-aware.** Each item is first classified into a *flow type*; a constraint only
  counts where it legitimately applies. A 2-way item with no goods receipt is not a "missing
  receipt" — that flow never expects one. This is the single biggest source of false alarms WISE
  removes.
- **Auditable by construction.** Every case score decomposes *exactly* into its layer contributions
  (`1 − S = Σ_λ Δ_λ`, verified to ~1e-16). No black-box number ever reaches a decision-maker.
- **Scale × severity, not severity alone.** Priority rewards deviation **that occurs at scale**, so a
  huge slice with a moderate gap can outrank a tiny slice with an extreme gap.
- **Decision-layer artefacts are versioned and governed.** Norms, weights and slice definitions live
  in explicit files, not buried magic numbers — so stakeholders can *negotiate* them.

> **One-sentence summary:** WISE turns a raw event log into a **bounded, auditable backlog of ranked,
> explainable improvement opportunities**, tailored to each stakeholder's priorities.

---

## 2. The conceptual model (vocabulary you need)

| Concept | Symbol | Meaning |
|---|---|---|
| **Case** | σ | The unit being scored (BPIC'19: a PO **line item**). |
| **Governance unit** | — | The unit work is assigned to (BPIC'19: a PO **header / document**). |
| **Flow type** | — | The execution model an item runs (BPIC'19: DF1, DF2, 2-way, Consignment). Fixes which constraints apply. |
| **Constraint** | c | A machine-checkable rule producing a **bounded violation** `ν_c(σ) ∈ [0,1]` (0 = satisfied, 1 = maximally violated). |
| **Applicability map** | C_app(σ) | The subset of constraints that legitimately apply to σ, given its flow type. |
| **Layer** | λ | A business-facing group of constraints. Layers *partition* the constraint set, giving interpretable "why". |
| **View** | p | A stakeholder priority = a weight vector over the shared norm. Same checks, different emphasis. |
| **Case score** | S⁽ᵖ⁾(σ) | Applicability-weighted fraction of the norm satisfied under view p (1 = perfect). |
| **Layer contribution** | Δ_λ⁽ᵖ⁾(σ) | How much layer λ contributes to the case's penalty. `1 − S = Σ_λ Δ_λ` exactly. |
| **Slice** | s | A business segment to prioritise, via `g(σ)` (BPIC'19: company × spend area). |
| **Priority Index** | PI⁽ᵖ⁾_s | Scale × underperformance of a slice. The backlog is ranked by this. |

**A process norm is a triple** `N = (C, Λ, lay)`:
- `C` — the finite set of constraints,
- `Λ` — the finite set of layers,
- `lay` — assigns each constraint to exactly one layer (so `{C_λ}` partitions `C`).

A **view** does *not* define a different norm — it defines different *priorities* over the same norm.
That is what lets one set of case evidence answer Finance, Logistics, Compliance, and Automation
questions without re-checking anything.

---

## 3. The WISE pipeline, step by step

The pipeline has **three phases** preceded by a **governance-framing** step and followed by a
**governance loop**. The BPIC'19 instantiation lives in the [wise_eval/](wise_eval/) package; the
narrative below maps each step to its module and to the EXTENDED notebook sections.

```
        Governance framing  →  Phase 1  →  Phase 2  →  Phase 3  →  Governance loop  →  (Root-cause)
        (case + flow types)    (norm)     (scoring)   (priority)   (qualify findings)    (Section 8)
```

### Step 0 — Governance framing (before any computation)
*Module:* [wise_eval/data.py](wise_eval/data.py) · *Notebook:* cells around §"event log and flow types"

Two decisions are made *before* any score exists, because they shape everything downstream:

1. **Fix the case notion and governance unit.** *What* gets scored (PO item) vs *what* improvement is
   assigned to (PO header). In WISE these can differ — you score fine-grained items but act on the
   accountable unit.
2. **Classify every item into a flow type.** The same event pattern is compliant in one flow and
   deviant in another, so flow typing is the *applicability backbone*. The BPIC'19 mapping is from
   `case Item Category` → {DF1 (3-way, invoice after GR), DF2 (3-way, invoice before GR), 2-way,
   Consignment}.

> **Why it matters:** mixing flows is the most common way dashboards invent false alarms. WISE
> conditions every later number on "what *should* have happened for *this* kind of item."

### Phase 1 — Norm definition: constraints, layers, views, weights
*Module:* [wise_eval/norm.py](wise_eval/norm.py), [wise_eval/constraints.py](wise_eval/constraints.py) · *Norm file:* [norm/bpic19_norm_v1.json](norm/) · *Notebook:* §1 "The norm in full"

**1a. Constraints → bounded violations `ν_c(σ) ∈ [0,1]`.** WISE uses a small catalogue of
parameterised constraint types (the paper's five types and their implementation):

| Paper type | Meaning | Implementation types |
|---|---|---|
| `pres` presence | an expected event must occur | `presence`, `presence_any` |
| `order` / `lag` | event B must follow A, optionally within a time window | `lag` (`order` = lag with δ=∆=0) |
| `sing` singularity | bounded number of repeats | `count_excess`, `threshold_excess` |
| `excl` exclusion | a forbidden event must not occur | `absence`, `absence_any` |
| `bal` balance | numeric values must match within tolerance | `networth_cv` proxies, `*_with_multiplier` |

All numeric deviations pass through the **threshold–saturation rule**, which gives a tolerance band
then a linear ramp to a hard cap — making counts, time lags and value mismatches comparable on one
[0,1] scale:

```
sat(z; ϑ, W) = 0                  if z ≤ ϑ          (tolerated, no penalty)
             = min((z − ϑ)/W, 1)  if z > ϑ          (linear penalty, caps at 1)
```
*Example:* ϑ=10, W=20 → sat(8)=0, sat(15)=0.25, sat(30)=1.

The BPIC'19 norm has **29 flow-aware constraints**.

**1b. Layers `Λ` — the "why" vocabulary.** Constraints are grouped into 7 business-facing layers.
Layers don't add checks; they let a slice be *explained* by which mechanisms dominate:

| Layer | Name | What it captures |
|---|---|---|
| **L1** | Closure & completeness | expected completion / closure (e.g. invoice & clearing present) |
| **L2** | Flow discipline | flow-conditioned ordering control (the right sequence for the flow type) |
| **L3** | Timeliness & ageing | hand-over lags and ageing (invoice→clear, goods→invoice) |
| **L4** | Rework & instability | repeats, fragmentation, churn |
| **L5** | Exceptions & corrections | payment blocks, manual corrections |
| **L6** | Value & commercial integrity | value/quantity mismatches, commercial soundness |
| **L7** | Effort & automation friction | manual touches, number of distinct humans, touchless rate |

**1c. Views `p` — stakeholder priorities.** A view is a vector of **layer weights** `a_λ⁽ᵖ⁾ ≥ 0`,
`Σ_λ a_λ = 1`. The four BPIC'19 views (each column of the norm's layer-weight table):

| Layer | Finance | Logistics | Compliance | Automation |
|---|---|---|---|---|
| L1 closure | **0.24** | 0.08 | 0.12 | 0.06 |
| L2 flow | 0.08 | **0.24** | **0.30** | 0.08 |
| L3 ageing | **0.24** | 0.22 | 0.10 | 0.10 |
| L4 rework | 0.06 | **0.22** | 0.06 | **0.28** |
| L5 exceptions | 0.08 | 0.08 | **0.26** | 0.06 |
| L6 value | **0.24** | 0.04 | 0.06 | 0.08 |
| L7 effort | 0.06 | 0.12 | 0.10 | **0.34** |

> The weight table is the **only** thing that differs between views; the constraints and violations
> are identical. This is the artefact you actually *negotiate* with stakeholders — change one weight
> and re-rank; you do not rebuild the analysis.

**1d. Two-stage weight elicitation.** Final raw constraint weights are built in two governed stages:
```
1. layer weights      a_λ⁽ᵖ⁾ ≥ 0,  Σ_λ a_λ⁽ᵖ⁾ = 1     (per view — the table above)
2. within-layer       b̂_c = b_c / Σ_{d∈C_λ} b_d         (shared across views)
   ⇒ raw weight       w_c⁽ᵖ⁾ = a_lay(c)⁽ᵖ⁾ · b̂_c
```
The scoring equations consume only `w_c⁽ᵖ⁾`; the two-stage split is for elicitation and governance.

### Phase 2 — Case-level scoring with exact layer evidence
*Module:* [wise_eval/scoring.py](wise_eval/scoring.py) · *Notebook:* §2 "Phase-2 evidence across all views"

Each applicable constraint is evaluated, then aggregated into an **applicability-aware case score**:

```
                Σ_{c∈C_app(σ)} w_c⁽ᵖ⁾ · ν_c(σ)
S⁽ᵖ⁾(σ) = 1 −  ───────────────────────────────
                Σ_{c∈C_app(σ)} w_c⁽ᵖ⁾
```

So `S = 0.8` means "the applicable constraints have a weighted-average violation of 0.2 under view
p." Items with no positively-weighted applicable constraint are **unscored** (excluded from
aggregation, not counted as perfect).

The penalty decomposes **exactly** into the seven layer terms:
```
1 − S⁽ᵖ⁾(σ) = Σ_λ Δ_λ⁽ᵖ⁾(σ)      (verified: max |Σ Δλ − (1−S)| ≈ 1e-16)
```
This exactness is what makes slice explanations trustworthy — nothing is lost or double-counted.

The notebook also reports four **view baselines** `μ̄⁽ᵖ⁾` (BPIC'19: ≈0.82–0.87). A slice is only a
"hotspot" if it is meaningfully *worse than its view's baseline* — that stops the org from chasing
normal background noise.

### Phase 3 — Slice & document prioritisation (the Priority Index)
*Module:* [wise_eval/prioritization.py](wise_eval/prioritization.py) · *Notebook:* §3–§5

Case scores roll up into a ranked backlog. The **Priority Index** rewards scale × underperformance:

```
PI⁽ᵖ⁾_s = n_s · ( μ̄⁽ᵖ⁾ − μ_s⁽ᵖ⁾ )_+        scale × underperformance (clipped at 0)
```

**Shrinkage stabilisation** prevents tiny noisy slices from dominating: small slices are pulled
toward the global mean (`γ` = pseudo-count, default 20 → a 20-case slice keeps half its observed gap):
```
μ̃_s = n_s/(n_s+γ) · μ_s  +  γ/(n_s+γ) · μ̄
PĨ_s = v_s · ( μ̄ − μ̃_s )_+          v_s = n_s (volume) or E_s (exposure/value); γ=0 recovers basic PI
```

Two roll-ups are produced:
- **Document backlog** (PO headers) — the Q3.1 "stand-out documents" answer. BPIC'19 result:
  **~2–3% of documents carry ~80% of priority mass; <10% carry 95%.** The campaign is *bounded* —
  a focused team working a couple of thousand documents captures most recoverable value.
- **Slice backlog** (company × spend area) — segments with accountable owners.

**Hotspot typology** (operational labels attached to the top of each backlog — they route the *type*
of fix, not the maths):

| Type | Signature | Implied fix |
|---|---|---|
| **Reservoir** | big volume × small gap | queue / SLA / backlog management |
| **Severity** | small volume × large gap | local policy/config review (don't re-engineer) |
| **Mechanism** | in between, one layer/constraint dominates | targeted redesign / RPA of that step |

**View comparison.** Because views can correlate on average yet diverge in the action tail, WISE
computes pairwise **Jaccard overlap** of the top-k shortlists and population score correlation.
Choosing "the" view is therefore a **governance decision** — a practical pattern is to run Compliance
and Automation in parallel and merge where the shortlists agree.

**Layer-delta profiles** (`slice_layer_deltas`) show, per slice, which layers are *more pronounced
than the global average* — the routing rule from "where" to "which kind of problem."

### Governance loop — qualify findings before acting
*Module:* [wise_eval/governance.py](wise_eval/governance.py) · *Notebook:* §7

The credibility firewall that separates **process problems** from **data artefacts**. Three
diagnostics per slice:

- **Right-censoring** — invoice-bearing items still open near the end of the observation window may
  simply lack time to finish (BPIC'19 window: 60 days). A hotspot whose gap **survives** excluding
  censored items is real; one whose gap **collapses** was a window artefact → demote.
- **Event replication** — items with too many events per *distinct timestamp* (>2) signal
  extraction-level duplication, not genuine repeated work.
- **Quarterly trend** — is the hotspot **chronic** (flat-high), **drifting** (rising), or an
  **end-of-window illusion** (only spikes in censoring-affected quarters)? Only act on the unshaded
  quarters.

Each slice gets a triage label: *robust → proceed*, *elevated replication → verify logging*,
*gap collapses → demote / data-quality ticket*.

### Section 8 — Root-cause & explanatory analysis (beyond prioritisation)
*Module:* [wise_eval/explain.py](wise_eval/explain.py) · *Notebook:* §8

WISE tells you **where** and **which layer**; Section 8 adds eight **descriptive** methods that drill
into a chosen hotspot to generate and rank *hypotheses* for a root-cause workshop (no causal claims):

| Method | Question it answers |
|---|---|
| Constraint prevalence | which exact *check* (not just layer) carries the penalty |
| Constraint co-occurrence (lift) | which deviations travel together → shared mechanism |
| Contrastive drivers (Cohen's d) | what measurably separates the hotspot from the rest of the log |
| Surrogate decision tree | human-readable threshold rules for a low score |
| Permutation importance | model-agnostic global score drivers |
| Resource hand-off matrix | coordination overhead / segregation-of-duties surface |
| Variant severity | which concrete execution sequences are worst |
| Value-phase profile | is the deviation in the financially material high-value tail? |

> These generate *and rank* hypotheses — they replace "let's stare at some traces" with "here are the
> three drivers most worth testing, ranked." They do **not** estimate causal effects.

---

## 4. What the BPIC'19 run concluded (worked example)

The five headline insights the EXTENDED notebook reaches — useful as a template for what a WISE
read-out looks like:

1. **The problem is a backlog, not an anomaly hunt.** L3 (closure/ageing) dominates every view; ~22.6%
   of invoice-bearing items never reach `Clear Invoice`. Frame the programme as "work down a backlog
   and tighten settlement timeliness," not "hunt for rare anomalies."
2. **Priority is extremely concentrated.** ~2–3% of documents carry ~80% of priority mass → a bounded,
   assignable campaign.
3. **Top hotspots are different problems in the same place.** `companyID_0000 × Packaging` is a
   *reservoir* (L3 ageing → queue/SLA work); `companyID_0000 × Logistics` is a *mechanism* (L4
   fragmentation → redesign the receiving step). Don't staff them as one project.
4. **The lens is a real decision.** Finance and Automation shortlists barely overlap in the action
   tail even when scores correlate.
5. **Some of the headline number is measurement, not process.** Part of the 22.6% is right-censoring;
   one slice (`companyID_0000 × (missing)` spend area) is a master-data gap, not a process defect.

---

## 5. How to run WISE on the BPIC'19 dataset

```bash
# 1. Environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Data — drop the CSV into data/ (any filename ending in .csv works)
#    data/BPI_Challenge_2019.csv

# 3. Run a notebook (Run All), in increasing depth:
#    WISE_BPIC19_basic_evaluation.ipynb       — compact method walkthrough
#    WISE_BPIC19_full_analysis.ipynb          — full multi-view analysis
#    WISE_BPIC19_full_analysis_EXTENDED.ipynb — same + an interpretation cell after every figure
#    WISE_BPIC19_paper_story.ipynb            — paper-figure reproduction (figures/paper/)
```

Without the CSV, the pipeline falls back to a **synthetic log with planted hotspots** so it still
demonstrates itself. All knobs live in [wise_eval/config.py](wise_eval/config.py) (`CONFIG`), the
single auditable place for every methodological choice. To point at a custom data folder without
editing the file:

```python
from wise_eval.config import CONFIG
CONFIG["csv_search_dirs"].insert(0, r"/path/to/my/data/folder")
```

**WISE Studio** (interactive Streamlit app): `streamlit run WISE_Studio/app.py` — upload & map
columns, build/load a norm, score, prioritise, and export.

---

## 6. How to apply WISE to a *different* dataset

WISE is process-agnostic; only the **norm**, the **column mapping**, and the **flow typing** are
domain-specific. Work through these steps in order.

### Step 1 — Frame the governance question
Decide the three things that anchor everything else:
- **Case σ** — the unit you score (an order line, a claim, a ticket, a patient episode).
- **Governance unit** — what improvement is assigned to (an order, a customer, a team).
- **Slicing `g(σ)`** — the segments you prioritise, ideally mapping to **accountable owners**
  (e.g. region × product, team × case-type). Keep this small — every extra key multiplies slices and
  erodes statistical support.

### Step 2 — Map your log columns
In `CONFIG` (or via WISE Studio's mapper), set the columns WISE relies on. The canonical names the
BPIC'19 pipeline expects are in `data.py`'s `REQUIRED_COLUMNS`; for a new log either rename your
headers to these or adapt the loader. The essentials:

| Role | BPIC'19 column | Your column |
|---|---|---|
| Case id | `case concept:name` | … |
| Governance unit | `case Purchasing Document` | … |
| Activity | `event concept:name` | … |
| Timestamp | `event time:timestamp` | … |
| Resource | `event org:resource` | … |
| Slice keys | `case Company`, `case Spend area text` | set `CONFIG["slice_keys"]` |
| Drill-down keys | `case Vendor`, `case Item Type`, quarter | set `CONFIG["drilldown_keys"]` |
| Flow-typing field | `case Item Category` | the attribute that determines applicability |
| (optional) Exposure/value | `event Cumulative net worth (EUR)` | for value-weighted PI |

### Step 3 — Define flow types (the applicability backbone)
Identify the **execution models** your process actually runs, where the same behaviour is legitimate
in one and deviant in another. Build the equivalent of `FLOW_MAP` (a mapping from a case attribute to
your flow types). If your process has only one execution model, every constraint is universally
applicable and you can skip this — but most real processes have several.

### Step 4 — Author the norm (`norm/<your_norm>.json`)
This is the main intellectual work. The norm JSON declares:
- **`constraints`** — each with an `id`, a `layer_id`, a `type` (presence / lag / order /
  count_excess / absence / balance …), the `applicable_flows` it applies to, `params` (activity
  names, thresholds ϑ, widths W, lag windows), and a `within_layer_weight` `b_c`.
- **`layers`** — your business-facing grouping. The 7 BPIC'19 layers (closure, flow, ageing, rework,
  exceptions, value, effort) are a strong general-purpose default; rename/regroup to fit your domain.
- **`views`** — one `layer_weights` vector per stakeholder (must sum to 1). Start with the four
  BPIC'19 roles and re-weight, or define your own.

Tips for writing constraints:
- Express each as a *bounded* signal; pick ϑ (tolerance) and W (ramp width) so `sat()` caps at a
  deviation you'd call "clearly bad."
- Tag each constraint with the flow types where it is legitimate — **getting applicability right is
  what removes false alarms.**
- Keep the catalogue small and machine-checkable. Layers, not constraint count, carry the
  interpretability.

Point `CONFIG["norm_path"]` at your file.

### Step 5 — Tune prioritisation & governance knobs
- `shrinkage_gamma` — raise it if you have many small noisy slices; lower toward 0 to trust observed
  gaps.
- `censoring_window_days` — set to your extraction window's tail so right-censoring is caught.
- `replication_ratio_flag` — events-per-distinct-timestamp threshold for your logging.
- `top_k_documents`, `pareto_threshold` — reporting/shortlist sizes.

### Step 6 — Run, then read with the governance loop
Run scoring → prioritisation → **always pass the backlog through the governance loop** before acting.
Use the layer-delta profile to route each hotspot to the right *kind* of fix, then (optionally) run
Section 8's drill-downs on the top hotspot to feed a root-cause workshop.

### Step 7 — Institutionalise
Freeze the norm, weights and slice definitions as a **versioned artefact**, re-run each period, and
track each top slice's stable gap over time. The quarterly-trend view then tells you whether an
intervention actually moved the chronic signal.

---

## 7. Where things live

| What | Path |
|---|---|
| Pipeline package (config, data, norm, scoring, prioritisation, governance, explain, viz) | [wise_eval/](wise_eval/) |
| All methodological knobs (the one place to configure) | [wise_eval/config.py](wise_eval/config.py) |
| Norm artefact (constraints, layers, views, weights) | [norm/](norm/) |
| Alternative library implementation | [src/wise/](src/wise/) |
| Notebooks (basic → full → extended → paper) | repo root, `WISE_BPIC19_*.ipynb` |
| Interactive app | [WISE_Studio/app.py](WISE_Studio/app.py) |
| Generated figures | [figures/](figures/) |
| Script entry point (uses `settings.json`) | [main.py](main.py) |

---

## 8. What WISE deliberately does *not* claim

WISE is **descriptive decision-support**. It does **not** estimate the causal effect of any
intervention, audit transactional correctness, or prove financial loss. High-priority slices are
**candidates** whose drivers are *ranked for investigation*; realised impact depends on feasibility
and root causes confirmed in follow-up. Always validate against system configuration, master data and
local process knowledge before committing change.
