import pandas as pd
from helper import get_category_data

class BaseScoreCalculator:
    def __init__(self, pivot_data, categories, cat, index='case_id'):
        self.pivot_data = pivot_data
        self.categories = categories
        self.cat = cat
        self.index = index

    def calculate_score(self):
        self.pivot_data['score'] = self.pivot_data.apply(self.score_calculation, axis=1)
        self.pivot_data['score'].fillna(0, inplace=True)
        
        if self.index in self.pivot_data.columns:
            grouped_scores = self.pivot_data.groupby(self.index)['score'].agg(['sum', 'count'])
            grouped_scores['normalized'] = grouped_scores['sum'] / grouped_scores['sum'].sum()
            return grouped_scores['normalized']
        else:
            print("The specified index column does not exist in pivot_data.")
            return self.pivot_data['score']

    def score_calculation(self, row):
        raise NotImplementedError("Subclasses should implement this!")