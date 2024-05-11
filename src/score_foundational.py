import pandas as pd
from helper import get_category_data

class ScoreFoundationalCalculator:
    def __init__(self, pivot_frequencies, categories, cat, index='case_id'):
        self.pivot_frequencies = pivot_frequencies
        self.categories = categories  # This should be the parsed YAML data under 'Categories'
        self.cat = cat
        self.index = index

    def calculate_score(self):
        # Calculate scores based on pivot_frequencies directly
        self.pivot_frequencies['score'] = self.pivot_frequencies.apply(self.score_found_calculation, axis=1)
        
        # Replace NaN scores with 0
        self.pivot_frequencies['score'].fillna(0, inplace=True)
        
        # Group scores by the 'index' column and sum them, then normalize
        if 'index' in self.pivot_frequencies.columns:
            # Calculate sum and count for each group
            grouped_scores = self.pivot_frequencies.groupby('index')['score'].agg(['sum', 'count'])
            
            # Normalize scores within each group: score / sum of scores in the group
            # This will create a normalized score column which sums up to 1 within each group
            grouped_scores['normalized'] = grouped_scores['sum'] / grouped_scores['sum'].sum()
            
            return grouped_scores['normalized']
        else:
            print("The specified index column does not exist in pivot_frequencies.")
            return self.pivot_frequencies['score']
    
    def score_found_calculation(self, row):
            category_data = get_category_data(self.categories, 'Foundational_Scores', row['category'])
            if not category_data:
                return None

            found_events = category_data['events']
            weights = category_data['weights']

            # Check that the number of weights matches the number of events
            if len(found_events) != len(weights):
                print("Warning: The number of weights must match the number of events")
                return 0

            # Compute the weighted score
            total_possible_score = sum(weights)
            actual_score = 0
            for event, weight in zip(found_events, weights):
                if row.get(event, 0) > 0:
                    actual_score += weight
                else:
                    # Subtract a fraction of its weight if an event is missing
                    actual_score -= (1 / len(weights)) * weight

            # Normalize the score
            normalized_score = actual_score / total_possible_score if total_possible_score > 0 else 0
            return normalized_score