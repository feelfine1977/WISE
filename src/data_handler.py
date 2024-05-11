import pandas as pd
import json
import pm4py
import yaml

class DataHandler:
    def __init__(self, MAC=True, DataName="BPIC_2019", layer="Generalized_Process_Standard_General", base_path=None, pivot_cat='case_id'):
        self.MAC = MAC
        self.DataName = DataName
        self.base_path = base_path if base_path else self._determine_base_path(MAC)
        self.data = None
        self.log = None
        self.layer = layer
        self.weights_yaml = None
        self.pivot_cat = pivot_cat
        self.json = None
        

    def _determine_base_path(self, MAC):
        return '/Users/urszulajessen/code/gitHub/WISE/data/' if MAC else 'C:\\Code\\Github\\WISE\\wise_flow\\data\\'

    def load_data(self):
        paths = self._get_data_paths()
        self.data = pd.read_csv(paths['csv'], sep=',')
        self._apply_reverse_mapping(paths['json'])
        self._convert_numerical_columns()
        self.data = pd.DataFrame(self.data)  # Ensure data is DataFrame
        self.log = self._format_log_data()
        self.weights_yaml = self._load_yaml(paths['weights_yaml'])
        return self.data,  self.weights_yaml, self.log, self._load_json(paths['json'])

    def _get_data_paths(self):
        base_path = f'{self.base_path}data_{self.DataName}/'
        return {
            'csv': f'{base_path}{self.DataName}.csv',
            'json': f'{base_path}{self.DataName}.json',
            'weights_json': f'{base_path}{self.DataName}_weights.json',
            'weights_yaml': f'{base_path}{self.DataName}_weights.yaml',
        }

    def _apply_reverse_mapping(self, json_path):
        with open(json_path) as file:
            data_desc = json.load(file)
        reverse_mapping = {v: k for k, v in data_desc.items() if v in self.data.columns}
        self.data.rename(columns=reverse_mapping, inplace=True)

    def _convert_numerical_columns(self):
        for column in self.data.columns:
            if column.startswith('num_'):
                try:
                    self.data[column] = pd.to_numeric(self.data[column].str.replace(',', '.'), errors='coerce')
                except AttributeError:
                    self.data[column] = pd.to_numeric(self.data[column], errors='coerce')

    def _format_log_data(self):
        return pm4py.format_dataframe(self.data, case_id='case_id', activity_key='activity', timestamp_key='timestamp')

        
    def _load_yaml(self, path):
        with open(path) as file:
             data = yaml.safe_load(file)
             for layer in data['Layers']:
                 if layer['Layer'] == self.layer:
                     return layer
                 
    def _load_json(self, path):
        with open(path) as file:
            return json.load(file)

    def pivot_table_frequencies(self, cat="cat_dim_6"):
        index = self.pivot_cat
        self.data['time:timestamp'] = pd.to_datetime(self.data['timestamp'])
        pivot = self.data.pivot_table(index='case_id', columns='activity', values='timestamp', aggfunc='size').fillna(0)
        pivot = self._add_index_column(pivot, index)
        return self._add_category_column(pivot, cat)

    def pivot_table_timestamps(self, cat="cat_dim_6"):
        index = self.pivot_cat
        self.data.sort_values(['case_id', 'timestamp', 'activity'], inplace=True)
        # Drop duplicate 'activity' for each 'case_id', keeping the first occurrence
        first_occurrences = self.data.drop_duplicates(subset=['case_id', 'activity'], keep='first')
        # Add a column that indicates the order of each activity within its case
        first_occurrences['activity_order'] = first_occurrences.groupby(index).cumcount() + 1
        # Pivot the table to have 'case_id' as index and 'activity' as columns,
        # values being the 'activity_order'
        pivot = first_occurrences.pivot(index='case_id', columns='activity', values='activity_order')
        
        pivot = pivot.fillna(0)
        pivot = self._add_index_column(pivot, index)
        return self._add_category_column(pivot, cat)
    


    def _add_category_column(self, pivot, cat):
        if cat:
            cat_series = self.data.drop_duplicates('case_id').set_index('case_id')[cat]
            pivot['category'] = pivot.index.map(cat_series)
        return pivot
    
    def _add_index_column(self, pivot, index):
        # Ensure `self.data` has 'case_id' set as an index for proper mapping
        if 'case_id' not in self.data.columns:
            print("Error: 'case_id' is not a column in the data")
            return pivot
        
        # Check if 'index' is a valid column in `self.data` and prepare to add it to the pivot
        if index in self.data.columns:
            # Ensure we drop duplicates and set 'case_id' as the index if not already set
            if self.data.index.name != 'case_id':
                indexed_data = self.data.drop_duplicates('case_id').set_index('case_id')
            else:
                indexed_data = self.data.drop_duplicates('case_id')
                
            if index != 'case_id':
                index_series = indexed_data[index]
            else:
                index_series = indexed_data.index.to_series()
            
            # Extract the series to add as a new column in the pivot table
            pivot['index'] = pivot.index.map(index_series)
        else:
            print(f"Warning: The specified index '{index}' is not available in data columns.")

        return pivot

