# WISE
Weighted Insights for Scoring Efficiency

WISE-Flow (Weighted Insights for Scoring Efficiency - FLOW)

## Introduction - Method Description
Anomalies in complex industrial processes are often obscured by high variability and complexity of event data, which hinders their identification and interpretation using process mining. To address this problem, we introduce WISE (Weighted Insights for Evaluating Efficiency), a novel method for analyzing business process metrics through the integration of domain knowledge, process mining, and machine learning.

The methodology involves defining business goals and establishing Process Norms with weighted constraints at the activity level, incorporating input from domain experts and process analysts. Individual process instances are scored based on these constraints, and the scores are normalized to identify features impacting process goals.

Evaluation using the BPIC 2019 dataset and real industrial contexts demonstrates that WISE enhances automation in business process analysis and effectively detects deviations from desired process flows. While LLMs support the analysis, the inclusion of domain experts ensures the accuracy and relevance of the findings.

## Installation

### Step 1: Create a New Conda Environment

To create a new conda environment for the WISE project, follow these steps:

1. **Open a terminal or command prompt.**

2. **Create a new conda environment:**

   ```sh
   conda create --name WISE python=3.8
   Activate the newly created environment:

   ```sh
    conda activate WISE
    ```
3. **Install the required packages:**

   ```sh
   pip install -r requirements.txt
   ```
4. **Add paths to your data in settings.json**
settings.json should be in the root folder of the project

   ```json
   {
    "data_path": "path/to/your/data",
    "reports_path": "path/to/your/reports",
    "log_path": "path/to/your/logs"
   }
   ```  
{
    "data_path": "path/to/your/data",
    "reports_path": "path/to/your/reports",
    "log_path": "path/to/your/logs"

}

## Phase 1
### Create Constraints and Weights (Optional only if you want to change the constraints and weights)
1. Define the constraints and weights for the process activities
Based on the file data/data_BPIC_2019/WISE_Framework_BPIC_2019.xlsx create or change the constraints and weights for the layers.
Example:
```
Undesirable Activity	Weight
Change Price	0,7
Change Quantity	0,6
Change Currency	0,75
Change Delivery Indicator	0,4
```
2. Convert the Excel file to a yaml file
Execute excel_to_yaml_converter.py

### Make sure you have following data in the data folder
data/data_BPIC_2019/BPIC_2019.csv - the event log from the BPIC 2019 challenge
data/data_BPIC_2019/BPIC_2019.json - mappings for the features that are to be used in the analysis
data/data_BPIC_2019/WISE_Framework_BPIC_2019.yaml - the constraints and weights for the process activities

### Run the WISE method
data/results/data_analysis.ipynb - the notebook that runs the WISE method
Caution in order to create new LLM report you have to provide OPENAI API key in the notebook!

## LLM Report
LLM Report can be found in the data/results folder
data/results
for generatin new LLM report you have to provide OPENAI API key in the notebook!
you need following fonts:
data/fonts/DejaVuSans-Bold.ttf
data/fonts/DejaVuSans.ttf
The fonts can be found in https://www.fontsquirrel.com/fonts/dejavu-sans
