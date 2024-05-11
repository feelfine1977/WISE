from helper import ScoreCalculator, get_category_data

class ScoreExclusionCalculator(ScoreCalculator):
    def __init__(self, pivot_frequencies, categories, cat, index = 'case_id'):
        self.pivot_frequencies = pivot_frequencies
        self.categories = categories  # This should be the parsed YAML data under 'Categories'
        self.cat = cat
        self.index = index


    def calculate_score(self):
        # Calculate scores based on pivot_frequencies directly
        self.pivot_frequencies['score'] = self.pivot_frequencies.apply(self.score_exclusion_calculation, axis=1)
        
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

    def score_exclusion_calculation(self, row):
        category_data = get_category_data(self.categories,'Exclusion_Scores', row['category'])
        if not category_data:
            return None

        exclusion_events = category_data['events']
        weights = category_data['weights']

        # Ensure that the number of weights matches the number of events
        if len(exclusion_events) != len(weights):
            raise ValueError("The number of weights must match the number of events")

        # Calculate weighted violations
        weighted_violations = sum(weight for event, weight in zip(exclusion_events, weights) if row.get(event, 0) > 0)

        total_weight = sum(weights)
        # Score is calculated based on the non-occurrence of weighted violations
        score = (total_weight - weighted_violations) / total_weight if total_weight > 0 else 0

        return score
