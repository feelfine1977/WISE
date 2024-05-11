from helper import ScoreCalculator, get_category_data

class ScoreSequentialCalculator(ScoreCalculator):
    def __init__(self, pivot_timestamps, categories, cat,index = 'case_id'):
        self.pivot_timestamps = pivot_timestamps
        self.categories = categories  # This should be the parsed YAML data under 'Categories'
        self.cat = cat
        self.index = index
    

    # def calculate_score(self):
    #     # Calculate scores based on pivot_timestamps directly
    #     scores = self.pivot_timestamps.apply(self.score_sequential_calculation, axis=1)
    #     return scores
    
    def calculate_score(self):
        # Calculate scores based on pivot_timestamps directly
        self.pivot_timestamps['score'] = self.pivot_timestamps.apply(self.score_sequential_calculation, axis=1)
        
        # Replace NaN scores with 0
        self.pivot_timestamps['score'].fillna(0, inplace=True)
        
        # Normalize scores between 0 and 1 based on the min and max of the calculated scores
        min_score = self.pivot_timestamps['score'].min()
        max_score = self.pivot_timestamps['score'].max()
        # Avoid division by zero if all scores are the same
        if max_score != min_score:
            self.pivot_timestamps['score'] = (self.pivot_timestamps['score'] - min_score) / (max_score - min_score)
        else:
            self.pivot_timestamps['score'] = 0  # Optional: Set all scores to zero if no variation
        
        # Group scores by the 'index' column and sum them, then normalize
        if 'index' in self.pivot_timestamps.columns:
            # Calculate sum for each group
            grouped_scores = self.pivot_timestamps.groupby('index')['score'].sum()
            
            # Normalize scores within each group: score / sum of scores in the group
            # This will create a normalized score column which sums up to 1 within each group
            grouped_scores = grouped_scores / grouped_scores.sum()
            
            return grouped_scores
        else:
            print("The specified index column does not exist in pivot_timestamps.")
            return self.pivot_timestamps['score']


    def score_sequential_calculation(self, row):
        category_data = get_category_data(self.categories,'Sequential_Scores', row['category'])
        if not category_data:
            return None

        sequential_events = category_data['events']
        weights = category_data['weights']

        # Ensure that the number of weights matches the number of event pairs
        if len(sequential_events) != len(weights):
            raise ValueError("The number of weights must match the number of event pairs")

        score = 0
        # Calculate scores with associated weights
        for event_pair, weight in zip(sequential_events, weights):
            if all(event in row for event in event_pair):
                if row[event_pair[0]] < row[event_pair[1]]:
                    score += weight  # The order is correct, add weight
                else:
                    score -= weight  # The order is incorrect, subtract weight

        # Normalize the score by the total of weights
        normalized_score = score / sum(weights) if weights else 0
        
        return normalized_score
