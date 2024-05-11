import pandas as pd
import numpy as np
import json

import pm4py
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interactive
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def get_data(MAC=True, Standard=True, DataName= "BPIC_2019"):
    if MAC:
        df_path = '/Users/urszulajessen/code/gitHub/WISE/data/'
    else:
        df_path = 'C:\Code\Github\WISE\wise_flow\data\BPI_Challenge_2019.csv'
        
    #path to the data folder = df_path + _data_ + DataName, csv_ data_folder_path + / + DataName + .csv json_ data_folder_path + / + DataName + .json
    path_data = df_path + 'data_' + DataName 
    path_csv = path_data + '/' + DataName + '.csv'
    path_json = path_data + '/' + DataName + '.json'
    path_json_weights = path_data + '/' + DataName + '_weights.json'
        
    data = pd.read_csv(path_csv, sep=',')
    
    
    #filter the irrelevant columns and rename remaining columns
    data_desc = json.load(open(path_json))
    reverse_mapping = {v: k for k, v in data_desc.items()}
    # Filter and rename the DataFrame
    filtered_data = data[list(reverse_mapping.keys())].rename(columns=reverse_mapping)
    #if column name begin with num_ then convert it to float
    for column in filtered_data.columns:
        if column.startswith('num_'):
            try:
                filtered_data[column] = filtered_data[column].str.replace(',', '.').astype(float)
            except:
                pass
    weights = json.load(open(path_json_weights))
    return filtered_data,weights


#Generalized Process Standard
#{Layer 1: Fouundational Protocols} consists of the generalized process norm focusing on the presence of essential, predefined events.
#{Layer 2: Sequential Directives} introduces $\Sigma \subseteq E \times E$, a set of ordered event pairs that must follow a mandatory sequence. For instance, if $(e_{\text{receive}}, e_{\text{pay}}) \in \Sigma$, receiving goods must precede the payment for those goods.
#{Layer 3: Equilibrium Standards} define a harmonization requirement across the process. Let $H \subseteq E$ denote the set of events that must be balanced. For example, if the process involves 50 purchase orders with 50 items, there should be an equal number of corresponding delivery and payment events, ensuring each item ordered is both delivered and paid for.
#{Layer 4: Singularity Criteria} specify events that should occur no more than once in a process instance. Let $U \subseteq E$ represent such unique events, deterring repetitions like double receiving goods or making duplicate payments.
#{Layer 5: Exclusion Guidelines}

#The weights.json file contains the weights for each category and the events that are relevant for each category.

def pivot_data(data):
    # Create a pivot table with case_id as index and event_name as columns,
    # the value is the number of times the event_name occurs in the case_id
    # sort data by case_id and event_time
    data = data.sort_values(['case_id', 'event_time'])
    pivot_data = data.groupby(['case_id', 'event_name']).size().unstack(fill_value=0)
    return pivot_data
    
def get_found_scores(pivoted_data, weights, case_id, category='Standard'):
    # Using the category to fetch relevant weights
    columns = weights.get(category, {}).get('found_events', [])
    columns_len = len(columns)
    score = 0
    for column in columns:
        if column in pivoted_data.columns:
            if pivoted_data.loc[case_id, column] > 0:
                score += 1
            else:
                score -= 1
    #Normalize the score and make them between 0 and 1
    score = (score + columns_len) / (2 * columns_len)

    return score


def check_event_sequence(group, seq):
    """ Helper function to check the first occurrence of events in sequence within grouped data, allowing for other events in between """
    # Assume 'group' is a DataFrame containing only the necessary 'event_name' column
    event_list = group.tolist()
    current_index = -1  # Start searching from the beginning
    for event in seq:
        try:
            new_index = event_list.index(event, current_index + 1)
            current_index = new_index
        except ValueError:
            return 0
    return 1

def get_sequence_scores(data, weights):
    # Sort data by case_id and event_time
    data = data.sort_values(['case_id', 'event_time'])

    # Group data by case_id, but make sure to use a slice of the DataFrame that only contains 'event_name' for simplicity
    grouped_data = data.groupby('case_id')['event_name']

    # Calculate the total number of sequences specified in weights
    total_sequences = sum(len(details['seq_events']) for details in weights.values())

    # Prepare a DataFrame to store the sequence scores for each case_id
    seq_scores = pd.DataFrame(index=grouped_data.groups.keys())  # Use the group keys as the index
    seq_scores['valid_sequences'] = 0  # Initialize the valid_sequences column

    # Check each sequence for each group
    for category, details in weights.items():
        for seq in details['seq_events']:
            # Apply the sequence check function to each group and accumulate the results
            seq_checks = grouped_data.apply(check_event_sequence, seq=seq)
            seq_scores['valid_sequences'] += seq_checks

    # Normalize the sequence score for each case_id
    if total_sequences > 0:
        seq_scores['seq_score'] = seq_scores['valid_sequences'] / total_sequences
    else:
        seq_scores['seq_score'] = 0  # Handle case where no sequences are defined

    # Merge the seq_scores back into the original data
    data = data.merge(seq_scores, left_on='case_id', right_index=True, how='left')

    return data

def calculate_equilibrium_score(data, weights):
    # Group data by case_id
    grouped_data = data.groupby('case_id')
    
    # Create a DataFrame to store the scores
    equilibrium_scores = pd.DataFrame(index=grouped_data.groups.keys())
    
    # Process each group
    for case_id, group in grouped_data:
        # Retrieve the category for this case to apply the correct weights
        # Assuming category can be determined directly or is uniform within a case
        category = group['cat_dim_5'].iloc[0]  
        
        # Get the equilibrium events for the current category
        equilibrium_events = weights.get(category, {}).get('equilibrium_events', [])
        
        # Count occurrences of each equilibrium event in the group
        event_counts = group[group['event_name'].isin(equilibrium_events)]['event_name'].value_counts()
        
        # Calculate the equilibrium score: 
        # 1 if all specified events occur the same number of times, else less than 1
        if len(event_counts) == len(equilibrium_events) and event_counts.nunique() == 1:
            score = 1.0
        else:
            # If not all events are present or they do not all occur the same number of times
            min_count = event_counts.min()
            max_count = event_counts.max()
            score = min_count / max_count if max_count > 0 else 0
        
        # Store the score
        equilibrium_scores.loc[case_id, 'equilibrium_score'] = score
    
    # Merge the equilibrium scores back into the original data
    data = data.merge(equilibrium_scores, left_on='case_id', right_index=True, how='left')
    
    return data

def calculate_singularity_score(data, weights):
    # Group data by case_id
    grouped_data = data.groupby('case_id')
    
    # Create a DataFrame to store the scores
    singularity_scores = pd.DataFrame(index=grouped_data.groups.keys())
    
    # Process each group
    for case_id, group in grouped_data:
        # Retrieve the category for this case to apply the correct weights
        # Assuming category can be determined directly or is uniform within a case
        category = group['cat_dim_5'].iloc[0]
        
        # Get the singular events for the current category
        singular_events = weights.get(category, {}).get('singular_events', [])
        
        # Count occurrences of each singular event in the group
        event_counts = group[group['event_name'].isin(singular_events)]['event_name'].value_counts()
        
        # Calculate the singularity score:
        # 1 if all specified singular events occur no more than once, otherwise 0
        if all(count <= 1 for count in event_counts):
            score = 1.0
        else:
            score = 0.0
        
        # Store the score
        singularity_scores.loc[case_id, 'singularity_score'] = score
    
    # Merge the singularity scores back into the original data
    data = data.merge(singularity_scores, left_on='case_id', right_index=True, how='left')
    return data
    
def calculate_exclusion_score(data, weights):
    # Group data by case_id
    grouped_data = data.groupby('case_id')
    
    # Create a DataFrame to store the scores
    exclusion_scores = pd.DataFrame(index=grouped_data.groups.keys())
    
    # Process each group
    for case_id, group in grouped_data:
        # Retrieve the category for this case to apply the correct weights
        category = group['cat_dim_5'].iloc[0]  # Assuming category can be determined from 'cat_dim_5'
        
        # Get the exclusion events for the current category
        exclusion_events = weights.get(category, {}).get('exclusion_events', [])
        
        # Check if any exclusion event is present in the group
        if group['event_name'].isin(exclusion_events).any():
            score = 0.0
        else:
            score = 1.0
        
        # Store the score
        exclusion_scores.loc[case_id, 'exclusion_score'] = score
    
    # Merge the exclusion scores back into the original data
    data = data.merge(exclusion_scores, left_on='case_id', right_index=True, how='left')
    
    return data

def main(DataName= "BPIC_2019"):
    # Load the data
    data,weights = get_data(MAC=True, Standard=False, DataName=DataName)  # Adjust this as per your data loading method
    pivoted_data = pivot_data(data)
    

    # Ensure there is a 'cat_dim_5' column in your DataFrame
    if 'cat_dim_5' not in data.columns:
        raise ValueError("cat_dim_5 column not found in the data")
    
    # Get score for each case_id based on its specific category in cat_dim_5
    data['score_found'] = data.apply(lambda row: get_found_scores(pivoted_data, weights, row['case_id'], row['cat_dim_5']), axis=1)
    data = get_sequence_scores(data, weights)    
    data = calculate_equilibrium_score(data, weights)
    data = calculate_singularity_score(data, weights)
    data = calculate_exclusion_score(data, weights)
    print(data.head())

if __name__ == '__main__':
    main("sample")

    