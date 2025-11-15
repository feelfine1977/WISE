# WISE

**W**eighted **I**nsights for **S**coring **E**fficiency  
(also used in some materials as *Weighted Insights for Evaluating Efficiency*)

WISE is a norm-based scoring method and Python library for process mining.  
It encodes business expectations as a set of constraints over event logs,
scores each case against those constraints, and aggregates scores over
business segments (“slices”) to indicate where deviations from the norm
are both frequent and pronounced.

The library is designed to be:

- **Modular** – layers (presence, order, balance, etc.) are pluggable.
- **Tool-agnostic** – works on standard case-centric event logs (CSV, XES → DataFrame).
- **Explainable** – scores decompose into per-constraint contributions.

The repository also contains an example analysis on the **BPIC 2019 P2P**
dataset and optional notebooks for further experiments.

---

## 1. Method overview (informal)

In many process mining projects, event logs are explored with dashboards,
generic PPIs, and ad-hoc rules. This often yields interesting diagnostics but
does not provide a small, auditable list of **where to act first**.

WISE takes a different route:

1. **Define a process norm**  
   Business goals (e.g. “reliable three-way matching”, “less rework”) are
   translated into a set of constraints grouped into *layers* such as:

   - *Presence* – mandatory steps should occur.
   - *Order/Lag* – steps should occur in the right order and within a lag.
   - *Balance* – quantities/amounts should “add up”.
   - *Singularity* – certain activities should not repeat.
   - *Exclusion* – forbidden steps or patterns should not occur.

   Each constraint can have different weights per *view* (e.g. Finance vs Logistics).

2. **Score cases against the norm**  
   For each case (trace), WISE computes a bounded violation for each
   constraint and aggregates them into a case score per view.

3. **Aggregate by slices and compute a Priority Index**  
   Cases are grouped into slices (e.g. `company × spend area × matching regime`).
   For each slice, WISE computes:

   - the average case score,
   - the gap to the global mean,
   - and a simple **Priority Index**: `PI = number_of_cases × gap`.

   This helps answer “which segments are furthest from the norm and affect the
   most cases?”.

WISE focuses on **descriptive** prioritisation. It does not estimate causal
effects of interventions; those remain the task of further analysis and
experimentation.

---

## 2. Repository structure

The core code is organised as a Python package:

```text
src/
  wise/
    __init__.py
    model.py        # Norm, Constraint, View data structures
    norm.py         # view-weight aggregation and helpers
    layers/         # pluggable layer implementations
      base.py       # BaseLayer interface
      presence.py
      order_lag.py
      balance.py
      singularity.py
      exclusion.py
    io/
      log_loader.py   # event log loading
      norm_loader.py  # norm loading (JSON, etc.)
    scoring/
      scoring.py    # case-level scores
      slices.py     # slice-level aggregation & PI
      eb.py         # empirical-Bayes shrinkage
      bootstrap.py  # optional confidence intervals
    ui/
      streamlit_app_placeholder.py  # placeholder for a future UI

tests/
  conftest.py
  test_presence_layer.py
  test_norm_loader.py
  test_scoring.py

data/
  BPIC_2019.csv         # example event log (not included in repo by default)
  WISE_norm.json        # example norm for BPIC 2019
  ...                   # optional: outputs (case scores, slice summary)

main.py                 # example script for running WISE on BPIC 2019
pyproject.toml          # build configuration
requirements.txt        # extra dependencies
README.md
```

## 3. Installation
### 3.1. Create and activate a virtual environment
You can use conda (shown here) or any other virtual env manager.

```bash
conda create -n wise-env python=3.10
conda activate wise-env
```

### 3.2. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```
The second command installs WISE as an editable package so you can import
wise from your own scripts and notebooks.

### 3.3 Run tests to verify installation
```bash
pytest
```
If tests pass, the core library is installed correctly.

## 4. Using WISE with BPIC 2019 (example)

### 4.1. Prepare data and a process norm

Place the following files in `data/`:

- `data/BPIC_2019.csv` – the BPIC 2019 P2P event log (download from the BPI challenge).
- `data/WISE_norm.json` – a norm definition for BPIC 2019.

`WISE_norm.json` should follow the JSON schema expected by
`wise.io.norm_loader`:

```json
{
  "views": ["Finance", "Logistics"],
  "constraints": [
    {
      "id": "c_l1_gr",
      "layer_id": "presence",
      "params": { "activity": "Record Goods Receipt" },
      "base_weight": 1.0,
      "view_weights": { "Finance": 0.2, "Logistics": 0.3 }
    }
    // ... more constraints ...
  ]
}
```

An example `WISE_norm.json` tailored to BPIC 2019 is already included in this
repository.

### 4.2. Run the example script

`main.py` contains a minimal example configured for BPIC 2019.

It expects the following BPIC 2019 column names:

```python
CASE_ID_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIMESTAMP_COL = "time:timestamp"
```

It uses the following slice attributes by default:

```python
SLICE_COLS = [
    "case_Company",
    "case_Spend_area_text",
    "case_Purch._Doc._Category_name",
]
```

To run the example:

```bash
python main.py
```

You should see in the terminal:

- the number of events and cases loaded,
- a preview of case scores,
- and a table of top slices by Priority Index.

The script also writes:

- `data/WISE_case_scores.csv` – case-level scores;
- `data/WISE_slice_summary.csv` – slice-level gaps and PIs.

You can adjust `LOG_PATH`, `NORM_PATH`, `SLICE_COLS`, and the view name at the
top of `main.py` to point to other logs or norms.

## 5. Using WISE as a library

You can also import WISE modules in your own scripts or notebooks. Example:

```python
import pandas as pd
from wise.io.log_loader import load_event_log
from wise.io.norm_loader import load_norm_from_json
from wise.scoring.scoring import compute_case_scores
from wise.scoring.slices import aggregate_slices

df = load_event_log(
    "data/BPIC_2019.csv",
    case_id_col="case:concept:name",
    activity_col="concept:name",
    timestamp_col="time:timestamp",
)

with open("data/WISE_norm.json", "r", encoding="utf-8") as f:
    norm = load_norm_from_json(f)

view_name = norm.get_view_names()[0]

case_scores = compute_case_scores(
    df=df,
    norm=norm,
    view_name=view_name,
    case_id_col="case:concept:name",
    activity_col="concept:name",
    timestamp_col="time:timestamp",
)

slice_summary = aggregate_slices(
    df_scores=case_scores,
    df_log=df,
    case_id_col="case:concept:name",
    slice_cols=["case_Company", "case_Spend_area_text"],
    shrink_k=50.0,
)
print(slice_summary.head())
```

## 6. Defining norms

Norms can be defined in JSON and loaded via `wise.io.norm_loader`. Each
constraint has:

- an `id` (string),
- a `layer_id` (e.g. `"presence"`, `"order_lag"`, `"balance"`),
- `params` (layer-specific configuration),
- `base_weight`,
- optional `view_weights` per view.

The available layers and their expected parameters are implemented in
`wise.layers.*`. You can add new layers by:

1. Creating a new module in `src/wise/layers/` that subclasses
   `BaseLayer` and implements `compute_violation`.

2. Registering it in `wise.layers.__init__` with a unique `LAYER_ID`.

## 7. Extending WISE

**New layers**  
Add a file in `wise/layers/`, implement `BaseLayer`, and register it in the
layer registry. Use the new `layer_id` in your norm JSON.

**New views**  
Add view names to the `views` list in the norm file, and specify
`view_weights` for constraints as needed.

**Streamlit UI**  
A placeholder UI module exists in `wise/ui/streamlit_app_placeholder.py`.
You can build a full Streamlit app that:

- uploads an event log (CSV),
- maps case / activity / timestamp columns,
- uploads a norm JSON,
- selects slices and a view,
- and displays case and slice summaries.

**Notebooks**  
You can place Jupyter notebooks under `notebooks/` and import `wise`
from there.

## 8. Status and caveats

WISE is currently a research/experimental codebase. It has been applied to
BPIC 2019 and a limited set of industrial P2P logs. The example norms and
weights included here should be seen as starting points, not ready-made
standards. For new contexts, norms should be adapted and reviewed with local
domain experts.

Contributions (bug reports, small PRs, or examples) are welcome.
