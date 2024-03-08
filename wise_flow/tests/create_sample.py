import pandas as pd
import pm4py
from pm4py.algo.filtering.log.variants import variants_filter



def create_sample(df, num_cases=100, sample_csv_path=None):
    """
    Create a sample of the original dataset
    """
    if sample_csv_path is None:
        sample_csv_path = 'C:\Code\Github\WISE\wise_flow\\tests\data/BPI_Challenge_2019_sample.csv'
    # Load the dataset
    dataframe = pm4py.format_dataframe(df, case_id='case_concept_name', activity_key='event_concept_name', timestamp_key='event_time_timestamp')
    sampled_df = pm4py.sample_cases(dataframe, num_cases, case_id_key='case:concept:name')
    sampled_df.to_csv(sample_csv_path, index=False)

    print(f"Sample CSV file created at: {sample_csv_path}")

    return sampled_df

dataframe = pd.read_csv('C:\Code\Github\WISE\wise_flow\data\BPI_Challenge_2019.csv', sep=',')
df = create_sample(dataframe, 1000)
