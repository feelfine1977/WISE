import pandas as pd
import pm4py
from pm4py.algo.filtering.log.variants import variants_filter



def create_sample(num_cases=100, sample_csv_path=None, Windows=True, version=None, create_csv=False):
    """
    Create a sample of the original dataset
    """
    if sample_csv_path is None and Windows:
        df = pd.read_csv('C:\Code\Github\WISE\wise_flow\data\BPI_Challenge_2019.csv', sep=',')
        sample_csv_path = 'C:\Code\Github\WISE\wise_flow\\tests\data/BPI_Challenge_2019_sample.csv'
    elif sample_csv_path is None and Windows is False:
        df = pd.read_csv('/Users/urszulajessen/code/gitHub/WISE/wise_flow/tests/data/BPI_Challenge_2019_sample.csv', sep=',')
        sample_csv_path = '/Users/urszulajessen/code/gitHub/WISE/wise_flow/data/BPI_Challenge_2019.csv'
    # Load the dataset
    dataframe = pm4py.format_dataframe(df, case_id='case_concept_name', activity_key='event_concept_name', timestamp_key='event_time_timestamp')

    dict = {'3-way match, invoice after GR':'DF1', '3-way match, invoice before GR':'DF2', '2-way match':'DF3', 'Consignment':'DF4'}
    #create a new column with the description of the item category
    dataframe['log_type'] = dataframe['case_Item_Category'].map(dict)
    if version is not None:
        dataframe = dataframe[dataframe['log_type'] == version]
    sampled_df = pm4py.sample_cases(dataframe, num_cases, case_id_key='case:concept:name')
    if create_csv:
        sampled_df.to_csv(sample_csv_path, index=False)
        print(f"Sample CSV file created at: {sample_csv_path}")

    return sampled_df

def get_eventlogs(dataframe, k=5, view='bpmn_graph', filtered=True, all_process_types=False, process_type=['Consignment']):
    eventlogs = {}
    dict = {'3-way match, invoice after GR':'DF1', '3-way match, invoice before GR':'DF2', '2-way match':'DF3', 'Consignment':'DF4'}
    if not all_process_types:
        #filter dict to only include the selected process types
        dict = {k: v for k, v in dict.items() if k in process_type}
    for key, value in dict.items():
        eventlogs['event_log_'+value] = pm4py.convert_to_event_log(dataframe[dataframe['log_type'] == value])
        if filtered:
            eventlogs['event_log_'+value] = pm4py.filter_variants_top_k(eventlogs['event_log_'+value], k=k)
        eventlogs['process_tree_'+value] = pm4py.discover_process_tree_inductive(eventlogs['event_log_'+value])
        eventlogs['bpmn_graph_'+value] = pm4py.convert_to_bpmn(eventlogs['process_tree_'+value])
        eventlogs['net_'+value], eventlogs['im_'+value], eventlogs['fm_'+value] = pm4py.convert_to_petri_net(eventlogs['bpmn_graph_'+value])
        eventlogs['dfg_'+value], eventlogs['sa_'+value], eventlogs['ea'+value] = pm4py.discover_directly_follows_graph(eventlogs['event_log_'+value])
        print(f"k={key}, v={value}, len={len(eventlogs['event_log_'+value])}")
        if view == 'bpmn_graph':
            pm4py.view_bpmn(eventlogs['bpmn_graph_'+value])
        elif view == 'petri_net':
            pm4py.view_petri_net(eventlogs['net_'+value], eventlogs['im_'+value], eventlogs['fm_'+value])
        elif view == 'dfg':
            pm4py.view_dfg(eventlogs['dfg_'+value], eventlogs['sa_'+value], eventlogs['ea'+value])
        else:
            pm4py.view_bpmn(eventlogs['bpmn_graph_'+value])

if __name__ == '__main__':
    WINDOWS = False
    VERSION = 'DF1'
    df_sample = create_sample(num_cases=1000, Windows=False, create_csv=False)
    len_variants = len(pm4py.get_variants(pm4py.convert_to_event_log(df_sample)))
                              
    top_variants = 5
    view_type = 'dfg'
    filtered_data = True
    all_process_types = False
    sel_process_types = ['3-way match, invoice after GR', '3-way match, invoice before GR']
    #get_eventlogs(df_sample, k=top_variants, view=view_type, filtered=filtered_data, all_process_types=False, process_type=sel_process_types)
    df = create_sample(num_cases=10000, Windows=WINDOWS, version=VERSION)
    print(len(df))