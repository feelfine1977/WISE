# WISE

**W**eighted **I**nsights for **S**coring **E**fficiency  
(also used as *Weighted Insights for Evaluating Efficiency*)

WISE is a norm-based scoring method and Python library for process mining.  
It encodes business expectations as a set of constraints over event logs,
scores each case against those constraints, and aggregates scores over
business segments (“slices”) to indicate where deviations from the norm
are both frequent and pronounced.

The library is designed to be:

- **Modular** – layers (presence, order, balance, etc.) are pluggable.
- **Tool-agnostic** – works on standard case-centric event logs (CSV → pandas).
- **Explainable** – scores decompose into per-constraint and per-layer contributions.
- **Interactive** – comes with a Streamlit UI to upload logs, build norms,
  run WISE, and explore heatmaps, boxplots, outlier views, and SHAP-based explanations.

---

## 1. Method overview (informal)

In many process mining projects, event logs are explored with dashboards,
generic PPIs, and ad-hoc rules. This often yields interesting diagnostics but
does not provide a small, auditable list of **where to act first**.

WISE takes a different route:

1. **Define a process norm**  
   Business goals (e.g. “reliable three-way matching”, “less rework”) are
   translated into a set of constraints grouped into layers such as:

   - *Presence* – mandatory steps should occur.
   - *Order/Lag* – steps should occur in the right order and within a time window.
   - *Balance* – quantities or amounts should “add up” within a tolerance.
   - *Singularity* – certain activities should not repeat.
   - *Exclusion* – forbidden steps or patterns should not occur.

   Each constraint can have different weights per *view* (e.g. Finance vs Logistics).

2. **Score cases against the norm**  
   For each case (trace), WISE computes a bounded violation for each
   constraint and aggregates them into a case-level score per view.

3. **Aggregate by slices and compute a Priority Index (PI)**  
   Cases are grouped into slices (e.g. `company × spend area × matching regime`).
   For each slice, WISE computes:

   - the average case score,
   - the gap to the global mean,
   - and a simple **Priority Index**  
     `PI = number_of_cases × gap`.

   This helps answer “which segments are furthest from the norm and affect the
   most cases?”.

WISE focuses on **descriptive** prioritisation. It does not estimate causal
effects of interventions; those remain the task of further analysis and
experimentation.

---

## 2. Mathematical foundation

### 2.1 Event log and constraints

Let:

- `L` be an event log containing events with case identifier, activity label and timestamp.
- For each case (trace) `σ` we have the ordered sequence of events for that case.
- `C` be the set of constraints in the process norm.

Each constraint `c ∈ C` has:

- a **layer** (presence, order_lag, balance, singularity, exclusion, …);
- a **parameter set** (e.g. activity name, from/to activities, max lag, tolerance);
- a **violation function**  
  `v_c(σ) ∈ [0, 1]`  
  where 0 = fully in norm, 1 = maximally violated for that constraint.

Examples:

- Presence:  
  $v_c(σ) = 0$ if activity $a$ occurs at least once in $σ$, 1 otherwise.
- Order/Lag:  
  $v_c(σ) = 0$ if $a_1$ occurs before $a_2$ and the lag is below a threshold;  
  gradually increasing towards 1 if $a_2$ is very late or missing.
- Balance:  
  $v_c(σ)$ proportional to the relative difference between quantities at two activities, capped at 1.
- Singularity:  
  $v_c(σ) = 0$ if an activity occurs at most once; higher values if it repeats.
- Exclusion:  
  $v_c(σ) = 1$ if a forbidden activity occurs, 0 otherwise.
### 2.2 Views and case scores

For each **view** `v` (Finance, Logistics, …) we assign non-negative weights
$w_c^{(v)}$ to each constraint $c$, typically normalised so that

$\sum_{c \in C} w_c^{(v)} = 1.$

The **case-level violation** in view $v$ is:

$\mathrm{Viol}^{(v)}(\sigma) = \sum_{c \in C} w_c^{(v)} \, v_c(\sigma).$

The **WISE case score** is then

$\boxed{S^{(v)}(\sigma) = 1 - \mathrm{Viol}^{(v)}(\sigma) \in [0, 1]}$

where 1 means fully in norm and 0 is maximally off-norm for the chosen
constraint set and weights.

WISE also retains per-layer and per-constraint violations for inspection:

- **Layer violations**: average violation per layer (presence, order, balance, …).
- **Constraint violations**: raw `v_c(σ)` per constraint `c`.

### 2.3 Slices and Priority Index

A **slice** `s` is a set of cases defined by attributes (e.g., company, spend area,
duration bucket, vendor). Let `|s|` be the number of cases in slice `s`.

For view `v`, define:

- Slice mean score:

  $\mu_s^{(v)} = \frac{1}{|s|} \sum_{\sigma \in s} S^{(v)}(\sigma).$

- Global mean score:

  $\bar{\mu}^{(v)} = \frac{1}{|L|} \sum_{\sigma \in L} S^{(v)}(\sigma).$
- Gap between slice and global mean:

  $\mathrm{gap}_s^{(v)} = \bar{\mu}^{(v)} - \mu_s^{(v)}.$

  Positive gap means slice `s` is **worse** than average (lower scores).

- **Priority Index (PI)**:

  $\boxed{\mathrm{PI}_s^{(v)} = |s| \cdot \mathrm{gap}_s^{(v)}}$

Slices with high positive $\mathrm{PI}_s^{(v)}$ aggregate large numbers of cases and large deviations, and are
therefore promising candidates for investigation.

### 2.4 Refinement targets

In the refinement step we use WISE outputs as **targets**:

- Overall score $S^{(v)}(\sigma)$ or its “badness” complement $1 - S^{(v)}(\sigma)$.
- Layer-level violations $violation\_layer$ (e.g. $violation\_presence$).
- Constraint-level violations $viol\_constraint\_id$ (e.g. $viol\_c001$).

These are used to:

- rank dimensions by how much they explain variation in the target,
- train a simple model and apply SHAP for local feature contributions,
- run outlier detectors and show unusual cases/slices.

---

## 3. Repository structure (simplified)

The core code is organised as a Python package:

```text
src/
  wise/
    __init__.py
    model.py           # Norm, Constraint, View data structures
    norm.py            # view-weight aggregation and helpers
    layers/            # pluggable layer implementations
      base.py          # BaseLayer interface
      presence.py
      order_lag.py
      balance.py
      singularity.py
      exclusion.py
    io/
      log_loader.py    # event log loading & normalisation
      norm_loader.py   # norm loading (JSON, etc.)
    scoring/
      scoring.py       # case-level scores + per-layer/constraint violations
      slices.py        # slice-level aggregation & PI, slice-layer matrices
      eb.py            # empirical-Bayes shrinkage
      bootstrap.py     # optional confidence intervals
    ui/
      app.py           # Streamlit entrypoint
      state.py         # shared session state helpers
      data_page.py     # upload & mapping UI
      norm_page.py     # interactive norm builder
      results_page.py  # scoring, heatmaps, boxplots
      refinement_page.py  # dimension ranking, SHAP & outliers

tests/
  conftest.py
  test_presence_layer.py
  test_norm_loader.py
  test_scoring.py

data/
  BPIC_2019.csv        # (optional) example event log – not in repo by default
  WISE_norm.json       # (optional) example norm
  ...                  # optional: outputs (case scores, slice summary, etc.)

pyproject.toml         # build configuration
requirements.txt       # dependencies
README.md
```

---

## 4. Installation

You need **Python 3.10+** and either `conda` or `venv`.  
The examples below use `conda`, but any virtual environment manager works.

### 4.1. Create and activate a virtual environment

```bash
conda create -n wise-env python=3.10
conda activate wise-env
```

### 4.2. Install dependencies and the package

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

The second command installs WISE as an editable package so you can import
`wise` from your own scripts, notebooks, or the Streamlit app.

### 4.3. Run tests (optional but recommended)

```bash
pytest
```

If tests pass, the core library is installed correctly.

---

## 5. Starting the Streamlit UI

The Streamlit app is the easiest way to use WISE:

```bash
conda activate wise-env
streamlit run src/wise/ui/app.py
```

Streamlit will print a local URL, typically:

```text
Local URL: http://localhost:8501
```

Open this in your browser.

---

## 6. Workflow inside the app

The app has **four** main pages (left sidebar):

1. **Data & Mapping**  
2. **Norm**  
3. **Results**  
4. **Refinement**  

### 6.1. Data & Mapping

Use this page to connect a CSV event log to WISE.

- Upload a CSV (e.g. BPIC 2019).
- Map:

  - Case ID → e.g. `case:concept:name`  
  - Activity → e.g. `concept:name`  
  - Timestamp → e.g. `time:timestamp`  

- Optionally derive extra slice dimensions:

  - day of week of first event (`case_dow_first`),
  - month of first event (`case_month_first`),
  - duration bucket (`case_duration_bucket`) based on a configurable threshold.

- Choose slice dimensions:

  - Manually (explicit multi-select), or  
  - Automatically: WISE selects low-cardinality columns (with configurable maximum cardinality and number of dimensions).

- Click **“Save dataset mapping”**.

**Interpretation tips**

- A **slice dimension** is any attribute you might want to group by later (company, spend area, vendor, duration bucket, month, …).
- Choosing too many slice dimensions at once can lead to a very large number of slices and slower computation. 2–4 core dimensions are often enough.

### 6.2. Norm

This page lets you **define and edit process norms**.

#### Views tab

- Views represent stakeholder perspectives, e.g. **Finance**, **Logistics**, **Automation**.
- You can add views or reset to defaults.
- You can also load an existing norm JSON; it will populate the builder and become the current norm.

#### Constraints tab

Build constraints per layer:

- **Presence (L1)** – activities that must occur at least once.  
- **Order/Lag (L2)** – “A then B within d days” expectations.  
- **Balance (L3)** – quantities (or amounts) that should match within a tolerance (e.g., GR vs INV quantities).  
- **Singularity (L4)** – activities that should not repeat.  
- **Exclusion (L5)** – forbidden activities.

The app proposes activity names and numeric columns from your data.  
A “Current constraints” table shows each constraint with:

- `id`,
- `layer_id`,
- a human-readable `description`,
- and raw `params`.

#### Weights & export tab

- A table with **constraints × views** lets you assign weights per view.
- Each cell is a discrete value in **0.0–1.0 (steps of 0.1)**:
  - `0.0` → this constraint does not contribute in that view,
  - `1.0` → full contribution,
  - intermediate values → intermediate importance.

- Click **“Use this norm in WISE”** to set it as the current norm.
- Click **“Download norm as JSON”** to export `WISE_norm.json` for reuse.

**Interpretation tips**

- Constraints describe **what good looks like** in process terms.
- Weights express **how much each rule matters** in a given view.
- All maths is linear and transparent; no hidden parameters.

### 6.3. Results

This page runs WISE scoring and shows slice-level overviews.

Steps:

1. Choose a **view** (e.g. Finance, Logistics).
2. Adjust **layer tuning sliders** (optional):
   - 0 → ignore that layer,
   - 1 → use view weights as defined in the norm,
   - >1 → boost that layer.
3. Choose **shrinkage k** (Empirical-Bayes):
   - Higher k pulls small slices more strongly toward the global mean,
   - This stabilises rankings against noise in small segments.
4. Click **“Compute scores and priorities”**.

Outputs:

- **Case-level scores (sample)**: each case’s WISE score in [0,1].
- **Slice-level table**:
  - `n_cases`, `mean_score`, `shrunk_mean_score`, `gap`, `PI`.
- **Top slices by PI** bar chart:
  - Bars to the right / red = slices that are worse than the norm,
  - Bars to the left / green = slices that outperform the norm.
- **Layer × slice heatmap**:
  - Rows: layers, columns: slices (full key or single dimension),
  - Colour = difference between slice’s average layer violation and global average (green = better, red = worse).
- **Constraint × slice heatmap**:
  - Similar, but rows are individual constraints within a chosen layer.
- **Boxplot by dimension**:
  - Shows distribution of case scores per category for a chosen dimension.
- **Scores heatmap by dimension**:
  - `mean_score` and layer-level scores per category (e.g., per spend area).

**Interpretation tips**

- Use PI to locate the **biggest levers** (many cases + bad scores).
- Use heatmaps to see which **layers or constraints** drive a slice’s badness.
- Use boxplots and per-dimension heatmaps to see within-slice variation and outliers.

### 6.4. Refinement

The **Refinement** page helps you move from “this slice is bad” to
“these are likely root causes”. It has three tabs:

1. **Global dimension ranking**  
2. **Local explanation (SHAP)**  
3. **Outliers (IF/LOF)**  

#### 6.4.1 Global dimension ranking

This tab uses WISE outputs as **targets** and ranks dimensions by how much they differentiate that target.

Targets can be:

- Score (lower = worse).
- Badness `1 - score` (higher = worse).
- Layer violations (e.g., `violation_presence`).
- Constraint violations (e.g., `viol_c001`).

For each dimension (e.g. spend area, vendor, duration bucket):

- Compute mean target per category.
- Compute `gap` = category mean – global mean.
- `max_abs_gap` = maximum absolute gap per dimension.

Dimensions are sorted by `max_abs_gap`. Higher values indicate dimensions with categories that are much better or worse than average.

**How to use**

- Use this tab to identify **high-value dimensions** to focus on (e.g. certain spend areas or vendor groups).
- Drill down on a dimension to see which categories are worst.

#### 6.4.2 Local explanation (SHAP)

This tab provides **feature-attribution explanations** of WISE targets.

Mathematically, we:

1. Build a feature matrix `X` from selected slice dimensions (company, spend area, vendor, duration bucket, month, …).
2. Train a simple model `f(X) ≈ target` (RandomForestRegressor).
3. Use SHAP to compute feature contributions `φ_j(x)` per case, such that roughly

   `f(x) - E[f(X)] ≈ Σ_j φ_j(x)`.

4. Interpret `φ_j(x)`:
   - `φ_j > 0`: feature `j` pushes the prediction **up** (towards higher badness/violation),
   - `φ_j < 0`: pushes it **down** (towards lower badness / closer to norm).

UI features:

- **Case ranking**: table of cases sorted by worst target.
- **Per-case SHAP**: bar chart of top features for a chosen case.
- **Slice-level SHAP**: for a chosen dimension and category (e.g., `case_dow_first = "Monday"`), average SHAP values across all sample cases in that category.

**How to use**

- Train the model once (configurable sampling size and features).
- For a bad case, inspect its top SHAP features to see which attributes (e.g. vendor, item category, month, duration bucket) explain its badness.
- For a bad slice (e.g. a specific spend area), use slice-level SHAP to see which attributes **systematically** drive its badness.

#### 6.4.3 Outliers (IsolationForest / Local Outlier Factor)

This tab uses unsupervised outlier methods to find **unusual cases and slices** based on features, not just on WISE scores.

Methods:

- **IsolationForest**: isolates anomalies by random partitioning; we use `-score_samples(X)` so higher is more outlying.
- **LocalOutlierFactor (LOF)**: compares local density; we use `-negative_outlier_factor_` so higher is more outlying.

For each case:

- Compute `outlier_score_norm` in [0,1].

For each category of a chosen dimension:

- Compute `mean_outlier` = average normalized outlier score per category,
- Also display `mean_target` (WISE target) for context.

**How to use**

- Use case-level outlier table to find **unusual cases** that may indicate data issues or rare patterns.
- Use scatter plot of outlier score vs WISE target to see which outliers are both unusual and bad.
- Use slice-level outlier view to find slices that have **many unusual cases**, even if their average WISE target is not yet terrible (potential early warning).

---

## 7. Using WISE programmatically

You can also import and use WISE in your own Python code, without the UI.

```python
import pandas as pd
from wise.io.log_loader import load_event_log
from wise.io.norm_loader import load_norm_from_json
from wise.scoring.scoring import compute_case_scores
from wise.scoring.slices import aggregate_slices

# 1. Load event log
df = load_event_log(
    "data/BPIC_2019.csv",
    case_id_col="case:concept:name",
    activity_col="concept:name",
    timestamp_col="time:timestamp",
)

# 2. Load process norm
with open("data/WISE_norm.json", "r", encoding="utf-8") as f:
    norm = load_norm_from_json(f)

view_name = norm.get_view_names()[0]  # e.g. "Finance"

# 3. Case-level scores
case_scores = compute_case_scores(
    df=df,
    norm=norm,
    view_name=view_name,
    case_id_col="case:concept:name",
    activity_col="concept:name",
    timestamp_col="time:timestamp",
)

# 4. Slice-level aggregation
slice_summary = aggregate_slices(
    df_scores=case_scores,
    df_log=df,
    case_id_col="case:concept:name",
    slice_cols=["case_Company", "case_Spend_area_text"],
    shrink_k=50.0,
)

print(slice_summary.head())
```

---

## 8. Defining norms via JSON

Norms can be defined in JSON and loaded via `wise.io.norm_loader`.

The Streamlit builder produces files that follow this schema:

```json
{
  "metadata": {
    "name": "P2P baseline norm"
  },
  "views": ["Finance", "Logistics"],
  "constraints": [
    {
      "id": "c001",
      "layer_id": "presence",
      "params": { "activity": "Record Goods Receipt" },
      "base_weight": 1.0,
      "view_weights": { "Finance": 0.2, "Logistics": 0.3 }
    },
    {
      "id": "c002",
      "layer_id": "order_lag",
      "params": {
        "activity_from": "Record Goods Receipt",
        "activity_to": "Record Invoice Receipt",
        "max_days": 10
      },
      "base_weight": 1.0,
      "view_weights": { "Finance": 0.3, "Logistics": 0.4 }
    }
    // ... more constraints ...
  ]
}
```

Each constraint has:

- `id` – unique identifier.
- `layer_id` – one of the registered layers
  (`presence`, `order_lag`, `balance`, `singularity`, `exclusion`, …).
- `params` – layer-specific configuration.
- `base_weight` – default importance.
- `view_weights` – optional overrides per view.

You can edit these JSON files by hand or use the Streamlit norm builder to
create and export them.

---

## 9. Extending WISE

- **New layers**  
  Add a file in `src/wise/layers/` that subclasses `BaseLayer` and implements
  `compute_violation(trace, constraint, activity_col, timestamp_col)`.  
  Register the layer’s `LAYER_ID` in `wise.layers.__init__` and use it in norms.

- **New views**  
  Add view names to the `views` list in the norm file, and specify
  `view_weights` for constraints as needed.

- **New visuals / analyses**  
  The Streamlit app is modular: you can add pages or sections that reuse
  `case_scores` and slice matrices for your own charts, or plug in additional
  XAI / anomaly detection methods.

---

## 10. Status and caveats

WISE is currently a research/experimental codebase. It has been applied to
BPIC 2019 and a limited set of industrial P2P logs. The example norms and
weights included here should be seen as starting points, not ready-made
standards. For new contexts, norms should be adapted and reviewed with local
domain experts.

Contributions (bug reports, small PRs, or examples) are very welcome.
