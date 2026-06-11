# WISE Studio — norm-based analysis for ANY event log

A generic, config-driven refactor of WISE. Point it at any CSV event log:
it detects your columns, guides you through building a **process norm**
(what good execution looks like), scores every case against it, and ranks
**business slices** into an actionable improvement backlog with hotspot types.

## Quickstart (app)

```bash
pip install -r requirements.txt
streamlit run app.py
```

1. **Upload & Map** — upload a CSV (or click *Use demo data*). WISE profiles the
   columns and suggests which is the case id / activity / timestamp / resource
   and which are business **dimensions** — confirm or override via dropdowns.
   Save the result as `log_mapping.json`.
2. **Build the Norm** — click *✨ Suggest a starter norm* (thresholds calibrated
   from your log's observed behaviour), then add/edit constraints of five types
   (presence, lag, singularity, exclusion, balance) with activity pickers and
   sliders; group them into layers and weight stakeholder **views**. Save as
   `norm.json`.
3. **Analyze** — pick slicing dimensions and run: baselines, ranked backlog with
   `reservoir / mechanism / severity` hotspot types, risk matrix, ownership
   heatmap, per-slice layer deltas, leading constraints, CSV/JSON downloads.

## Quickstart (headless pipeline)

The app is a thin layer — the same analysis runs without any UI:

```bash
python -m wise.pipeline events.csv log_mapping.json norm.json \
    --dims region product_type --out output
```

or programmatically:

```python
from wise.pipeline import run
res = run("events.csv", "log_mapping.json", "norm.json",
          slice_dims=["region", "product_type"], out_dir="output")
res["backlogs"]["Default"].head()
```

Figures land in `output/figures/`, tables in `output/tables/`.

## The two config files

**`log_mapping.json`** — what your columns mean:
```json
{
  "case_id": "order_id",
  "activity": "step",
  "timestamp": "when",
  "resource": "handler",
  "dimensions": ["region", "product_type"],
  "numeric_attributes": ["amount_eur"],
  "document_id": null,
  "encoding": "utf-8",
  "timestamp_format": null
}
```

**`norm.json`** — what good execution looks like (layers → constraints → views):
```json
{
  "layers": {"L1": {"name": "Completion & closure"}, "L2": {"name": "Timeliness"}},
  "constraints": [
    {"id": "c_presence_payment", "layer": "L1", "type": "presence",
     "activity": "Receive Payment", "min_count": 1},
    {"id": "c_lag_ship_invoice", "layer": "L2", "type": "lag",
     "from_activity": "Ship Order", "to_activity": "Send Invoice",
     "threshold_days": 3, "saturation_days": 7,
     "applicability": [{"column": "region", "values": ["North", "South"]}]}
  ],
  "views": {"Finance": {"layer_weights": {"L1": 0.6, "L2": 0.4}}}
}
```

Constraint types: `presence` (activity must occur), `lag` (B within N days of A),
`singularity` (bounded repetitions), `exclusion` (must not occur), `balance`
(two numeric totals match within tolerance). `applicability` restricts a check
to matching cases (e.g. one flow type) — non-matching cases are *not applicable*,
not penalised.

## BPIC'19 example

`examples/bpic19_mapping.json` + `examples/bpic19_norm.json` reproduce a compact,
flow-aware P2P norm (flow types via `case Item Category` applicability):

```bash
python -m wise.pipeline BPI_Challenge_2019.csv \
    examples/bpic19_mapping.json examples/bpic19_norm.json \
    --dims "case Company" "case Spend area text" --out output_bpic19
```

## Project layout

```
WISE_Studio/
├── app.py            ← Streamlit wizard (3 steps)
├── wise/             ← headless core (fully testable without the UI)
│   ├── mapping.py    ← schema, auto-detection, validation
│   ├── dataio.py     ← tolerant CSV load + canonicalisation
│   ├── features.py   ← generic case features
│   ├── constraints.py← 5 constraint types + threshold–saturation + applicability
│   ├── norm.py       ← norm schema, validation, suggest_norm()
│   ├── scoring.py    ← view scores + exact layer decomposition
│   ├── prioritize.py ← stable PI backlog, hotspot typing, layer deltas
│   ├── viz.py        ← figures (Streamlit- and file-friendly)
│   ├── pipeline.py   ← one-call run() + CLI
│   └── demo.py       ← built-in demo log with planted hotspots
└── examples/         ← BPIC'19 mapping + norm
```

## Method in one paragraph

A **norm** is a small set of machine-checkable constraints grouped into
business-facing **layers**; **views** weight the same constraints per
stakeholder. Each case gets a bounded score in [0,1] whose penalty decomposes
*exactly* into layer contributions. Cases aggregate into **slices** (your
dimensions) ranked by a shrinkage-stabilised **Priority Index**
(volume × underperformance), typed as *reservoir / mechanism / severity*
hotspots — each implying a different intervention. All outputs stay descriptive
(prioritisation and explanation, not causal claims).
