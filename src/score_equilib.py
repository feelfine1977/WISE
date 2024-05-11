from helper import ScoreCalculator, get_category_data

class ScoreEquilibCalculator(ScoreCalculator):
    def __init__(self, pivot_frequencies, categories, cat, index='case_id'):
        self.pivot_frequencies = pivot_frequencies
        self.categories = categories
        self.cat = cat
        self.index = index

    def calculate_score(self):
        # Calculate scores based on pivot_frequencies directly
        self.pivot_frequencies['score'] = self.pivot_frequencies.apply(self.score_equilib_calculation, axis=1)
        
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

    def score_equilib_calculation(self, row):
        category_data = get_category_data(self.categories, 'Equilibrium_Scores', row['category'])
        if not category_data:
            return None

        equilib_events = category_data['events']
        weights = category_data.get('weights', [1] * len(equilib_events))  # Default to equal weights if none provided

        # Fetch frequencies of each event, defaulting to 0 if not present
        event_counts = {event: row.get(event, 0) for event in equilib_events}

        if len(equilib_events) != len(weights):
            raise ValueError("The number of weights must match the number of events")

        # Calculate the total occurrences weighted by the predefined ratios
        total_occurrences = sum(event_counts[event] * weight for event, weight in zip(equilib_events, weights))
        total_expected = sum(weights)  # This represents the sum of all expected occurrences if all events occurred perfectly as per weights

        # Calculate the variance from expected ratios
        if total_expected == 0:
            return 0  # Avoid division by zero; handle cases where expected totals are zero

        # Normalize the total occurrences by the expected occurrences to get a score
        score = total_occurrences / total_expected
        # Ensure score is capped at 1.0
        return min(score, 1.0)
