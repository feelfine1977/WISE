import pandas as pd
import pm4py
from pm4py.algo.filtering.log.variants import variants_filter



def create_sample_log(df, num_cases=100, sample_csv_path=None, xes_path=None):
    """
    Create a sample of the original dataset
    """
    if sample_csv_path is None:
        sample_csv_path = 'C:\Code\Github\WISE\wise_flow\\tests\data/BPI_Challenge_2019_sample.csv'
    # Load the dataset
    dataframe = pm4py.format_dataframe(df, case_id='case_concept_name', activity_key='event_concept_name', timestamp_key='event_time_timestamp')
    sampled_log = pm4py.sample_cases(dataframe, num_cases, case_id_key='case:concept:name')
    if xes_path is not None:
        pm4py.write_xes(sampled_log, xes_path)
        print(f"Sample XES file created at: {xes_path}")
    if sample_csv_path is not None:
        sampled_log.to_csv(sample_csv_path, index=False)

    print(f"Sample CSV file created at: {sample_csv_path}")

    return sampled_log


if __name__ == "__main__":
    csv_path = "C:\Code\Github\WISE\wise_flow\\tests\data/BPI_Challenge_2019_sample.csv"
    xes_path = "C:\Code\Github\WISE\wise_flow\\tests\data/BPI_Challenge_2019_sample.xes"
    dataframe = pd.read_csv('C:\Code\Github\WISE\wise_flow\data\BPI_Challenge_2019.csv', sep=',')
    log = create_sample_log(dataframe, 1000, csv_path, xes_path)
