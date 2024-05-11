from helper import ScoreCalculator, get_category_data

class ScoreSingularCalculator(ScoreCalculator):
    def __init__(self, pivot_frequencies, categories, cat, index='case_id'):
        self.pivot_frequencies = pivot_frequencies
        self.categories = categories  # Assuming this is the parsed YAML data under 'Categories'
        self.cat = cat
        self.index = index


    def calculate_score(self):
        # Calculate scores based on pivot_frequencies directly
        self.pivot_frequencies['score'] = self.pivot_frequencies.apply(self.score_singular_calculation, axis=1)
        
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

    def score_singular_calculation(self, row):
        category_data = get_category_data(self.categories,'Singular_Scores', row['category'])
        if not category_data:
            return None

        singularity_events = category_data['events']
        weights = category_data['weights']

        # Ensure that the number of weights matches the number of events
        if len(singularity_events) != len(weights):
            raise ValueError("The number of weights must match the number of events")

        # Track the number of weighted singular violations
        violations = 0
        weighted_violations = 0

        for event, weight in zip(singularity_events, weights):
            if row.get(event, 0) > 1:
                violations += 1
                weighted_violations += weight  # Accumulate weighted violations

        total_weight = sum(weights)
        # If there are no singularity events specified, return a perfect score (1)
        if total_weight == 0:
            return 1.0

        # Score is calculated as the ratio of the sum of weights minus weighted violations to the total weights
        score = (total_weight - weighted_violations) / total_weight
        return score
