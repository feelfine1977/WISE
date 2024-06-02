from helper import exception_handler
from data_handler import DataHandler
import pandas as pd
from score_foundational import ScoreFoundationalCalculator
from score_sequential import ScoreSequentialCalculator
from score_equilib import ScoreEquilibCalculator
from score_singular import ScoreSingularCalculator
from score_exclusion import ScoreExclusionCalculator

class ProcessScoreCalculator:
    def __init__(self, data, weights_yaml, json,pivot_frequencies, pivot_timestamps, cat='cat_dim_6', index='case_id', grouped=False,view=None):
        self.data = data
        self.weights = weights_yaml
        self.cat = cat
        self.index = index
        self.grouped = grouped  # Flag to indicate if the data is already grouped
        self.json = json
        self.view = view

        # Initialize score calculators with the appropriate pivot table
        self.score_calculators = {
            'found': ScoreFoundationalCalculator(pivot_frequencies, self.weights, cat, index=self.index),
            'sequential': ScoreSequentialCalculator(pivot_timestamps, self.weights, cat, index=self.index),
            'equilibrium': ScoreEquilibCalculator(pivot_frequencies, self.weights, cat, index=self.index),
            'singular': ScoreSingularCalculator(pivot_frequencies, self.weights, cat, index=self.index),
            'exclusion': ScoreExclusionCalculator(pivot_frequencies, self.weights, cat, index=self.index)
        }
        self.calculate_scores()
        self.add_mean_score()
        
    def add_mean_score(self):
        self.data['mean_score'] = self.data.filter(like='score').mean(axis=1)

    def calculate_scores(self):
        score_frames = []
        # Iterate through each type of score calculator and merge their results into the main DataFrame
        for score_type, calculator in self.score_calculators.items():
            calculated_scores = calculator.calculate_score()
            calculated_scores = calculated_scores.to_frame(name=f'score_{score_type}_{self.index}')
            # Use provided index to merge scores back into the main DataFrame but only if the grouped flag is false otherwise the scores are already grouped
            if self.grouped:
                score_frames.append(calculated_scores)
            else:
                if self.index in self.data.columns:
                    self.data = pd.merge(self.data, calculated_scores, left_on=self.index, right_index=True, how='left')
                else:
                    self.data.set_index(self.index, inplace=True)
                    self.data = self.data.join(calculated_scores).reset_index()
            if self.grouped:
                all_scores = pd.concat(score_frames, axis=1)
                
                
                self.data = all_scores

def prepare_calculations(MAC=False, DataName="sample", layer=None, index='case_id', grouped=False,view=None):
    data_handler = DataHandler(MAC=MAC, DataName=DataName, layer=layer, pivot_cat=index)
    data, weights_yaml, log, json_file = data_handler.load_data()
    pivot_table_frequence = data_handler.pivot_table_frequencies(cat='cat_dim_6')
    pivot_table_timestamps = data_handler.pivot_table_timestamps(cat='cat_dim_6')
    score_calculator = ProcessScoreCalculator(data, weights_yaml, json=json_file, pivot_frequencies=pivot_table_frequence, 
                                              pivot_timestamps=pivot_table_timestamps, index=index,grouped=grouped,view=view)
    
    return score_calculator

if __name__ == '__main__':
    result = prepare_calculations(MAC=False, DataName="BPIC_2019", layer="General_Process_Standards", index='cat_dim_5', grouped=True)
    print(result.data)  # Optionally print or log the result for debugging
