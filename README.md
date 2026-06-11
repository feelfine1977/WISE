# WISE for BPIC 2019

WISE (Weighted Insights for Evaluating Efficiency) is a norm-based process analytics approach for identifying where process performance is weakest, why it is weak, and where improvement effort should be prioritized.

This repository contains:

- WISE analysis notebooks for BPIC 2019
- A Python implementation of scoring and prioritization logic
- WISE Studio, a Streamlit app for interactive analysis

## What WISE is

WISE evaluates event-log cases against an explicit process norm.

- A norm defines expected behavior through constraints
- Constraints are grouped into layers (for interpretability)
- Views apply stakeholder-specific layer weights (for decision context)
- Case scores are aggregated into slice-level priorities

In short, WISE turns raw process data into ranked, explainable improvement opportunities.

## WISE phases

The workflow in this project follows these phases:

1. Data preparation
   Load and standardize event logs, identify core columns (case id, activity, timestamp), and define slicing attributes.
2. Norm definition
   Load or design the process norm (constraints, layers, views, and weights).
3. Case-level scoring
   Evaluate each case against the norm to compute performance scores and layer-level contributions.
4. Slice prioritization
   Aggregate case scores by business slices and compute a Priority Index to identify high-impact hotspots.
5. Interpretation and actionability
   Use diagnostics, visualizations, and breakdowns to inform interventions and governance.

## Repository highlights

- Notebook analyses
  - WISE_BPIC19_basic_evaluation.ipynb
  - WISE_BPIC19_full_analysis.ipynb
  - WISE_BPIC19_paper_story.ipynb
- Core implementation
  - wise_eval/
  - src/wise/
- Interactive app
  - WISE_Studio/app.py
- Example data and norms
  - data/
  - norm/

## Installation

Use Python 3.10+ (recommended) and create a virtual environment.

### 1) Install core project requirements (notebooks + WISE analysis)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Install WISE Studio requirements

From the repository root:

```bash
pip install -r WISE_Studio/requirements.txt
```

## Using the WISE BPIC notebooks

1. Place the BPIC CSV in data/ (for example data/BPI_Challenge_2019.csv).
2. Check settings.json and confirm paths/columns:
   - data_path
   - norm_path
   - case_id_col
   - activity_col
   - timestamp_col
   - default_view
   - SLICE_COLS
3. Open one of the notebooks and run all cells.

Recommended order:

1. WISE_BPIC19_basic_evaluation.ipynb for a compact method walkthrough.
2. WISE_BPIC19_full_analysis.ipynb for full multi-view analysis.
3. WISE_BPIC19_paper_story.ipynb for paper-oriented figures and narrative.

Generated artifacts are written to figures/ and data/ (for example case and slice score CSVs).

## Running WISE Studio (Streamlit)

From the repository root:

```bash
streamlit run WISE_Studio/app.py
```

If streamlit is not found, ensure your virtual environment is activated and WISE_Studio dependencies are installed.

Typical WISE Studio flow:

1. Upload and map log columns
2. Build or load a norm
3. Run scoring and slice prioritization
4. Inspect hotspot diagnostics and export outputs

## Optional script execution

Run the pipeline script directly:

```bash
python main.py
```

This uses settings.json to load log and norm paths, compute case scores, aggregate slices, and export outputs under data/.

## License

See LICENSE.
