from helper import exception_handler
from data_handler import DataHandler

class ProcessScoreCalculator:
    def __init__(self, data, weights, log, cat='cat_dim_6'):
        self.data = data
        self.weights = weights
        self.log = log
        self.cat = cat

        # Calculate the score for the foundational events
        self.calculate_found_scores()

    def score_found_calculation(self, row):
        weights_cat = self.data[self.data.case_id == row.name][self.cat].iloc[0]
        found_events = self.weights[weights_cat]['found_events']
        
        # Calculate score based on whether the activity from found_events is greater than 0
        # Normalize the score by dividing by the number of found events, or 1 if there are none to avoid division by zero
        if len(found_events) == 0:
            normalized_score = 0  # If there are no events, the score is 0 as there's nothing to calculate.
        else:
            total_score = sum(1 if (activity in row and row[activity] > 0) else 0 for activity in found_events)
            normalized_score = total_score / len(found_events)
        
        return normalized_score
    
    def score_sequential_calculation(self, row):
        weights_cat = self.data[self.data.case_id == row.name][self.cat].iloc[0]
        sequential_events = self.weights[weights_cat]['seq_events']
        
        score = 0
        total_sequences = len(sequential_events)
        
        for event_pair in sequential_events:
            # Check if both events in the pair are in the row's index (activities)
            if all(event in row for event in event_pair):
                # Check if the first event's order is less than the second event's order
                if row[event_pair[0]] < row[event_pair[1]]:
                    score += 1  # The order is correct
                else:
                    score -= 1  # The order is incorrect

        # Normalize the score by the number of sequences if there are any sequences
        normalized_score = score / total_sequences if total_sequences > 0 else 0
        
        return normalized_score
    
    def score_equilib_calculation(self, row):
        weights_cat = self.data[self.data.case_id == row.name][self.cat].iloc[0]
        equilib_events = self.weights[weights_cat]['equilibrium_events']

        # Fetch frequencies of each event in equilibrium_events, defaulting to 0 if not present
        event_counts = {event: row.get(event, 0) for event in equilib_events}
        
        # If the list is empty or has only one type of event, it's trivially in equilibrium
        if len(event_counts) <= 1:
            return 1.0

        # Compute the minimum and maximum frequencies among the events
        min_freq = min(event_counts.values())
        max_freq = max(event_counts.values())

        # If max frequency is zero, which means none of the events occurred, return a score of 0
        if max_freq == 0:
            return 0

        # Score calculation: ratio of min frequency to max frequency
        # This ratio will be 1.0 if all frequencies are equal (perfect equilibrium)
        # and less than 1 as the variation between frequencies increases
        score = min_freq / max_freq

        return score
    
    def score_singularity_calculation(self, row):
        weights_cat = self.data[self.data.case_id == row.name][self.cat].iloc[0]
        singularity_events = self.weights[weights_cat]['singular_events']

        # Track the number of singular events that exceed the occurrence limit (which is 1)
        violations = 0
        total_singular = len(singularity_events)

        for event in singularity_events:
            # Count the occurrences of each singularity event; assume 0 if not present in the row
            if row.get(event, 0) > 1:
                violations += 1

        # If there are no singularity events specified, return a perfect score (1)
        if total_singular == 0:
            return 1.0

        # Score is calculated as the ratio of non-violated events to the total singularity events
        score = (total_singular - violations) / total_singular
        return score

    def score_exclusion_calculation(self, row):
        weights_cat = self.data[self.data.case_id == row.name][self.cat].iloc[0]
        exclusion_events = self.weights[weights_cat]['exclusion_events']

        # Check if any of the exclusion events occur
        violations = sum(1 for event in exclusion_events if row.get(event, 0) > 0)
        total_exclusions = len(exclusion_events)

        # If no exclusion events are specified, return a perfect score (1.0)
        if total_exclusions == 0:
            return 1.0

        # If any exclusion event occurs, it results in a violation, thus a score of 0.
        # If none of the specified exclusion events occur, the score is 1.0.
        score = 0.0 if violations > 0 else 1.0
        return score

    def pivot_data_frequence(self):
        self.data = self.data.sort_values(['case_id', 'timestamp'])
        pivot = self.data.groupby(['case_id', 'activity']).size().unstack(fill_value=0)
        # Calculate score for each row of the pivot table
        pivot['score_found'] = pivot.apply(lambda row: self.score_found_calculation(row), axis=1)
        #apply score to self.data on case_id
        self.data['score_found'] = self.data['case_id'].apply(lambda x: pivot.loc[x]['score_found'])
        
        pivot['score_equilib'] = pivot.apply(lambda row: self.score_equilib_calculation(row), axis=1)
        #apply score to self.data on case_id
        self.data['score_equilib'] = self.data['case_id'].apply(lambda x: pivot.loc[x]['score_equilib'])
        
        pivot['score_singularity'] = pivot.apply(lambda row: self.score_singularity_calculation(row), axis=1)
        #apply score to self.data on case_id
        self.data['score_singularity'] = self.data['case_id'].apply(lambda x: pivot.loc[x]['score_singularity'])
        
        pivot['score_exclusion'] = pivot.apply(lambda row: self.score_exclusion_calculation(row), axis=1)
        #apply score to self.data on case_id
        self.data['score_exclusion'] = self.data['case_id'].apply(lambda x: pivot.loc[x]['score_exclusion'])
        
    def pivot_data_timestamps(self):
        # Sort the data by 'case_id', 'timestamp', and 'activity'
        sorted_data = self.data.sort_values(['case_id', 'timestamp', 'activity'])

        # Drop duplicate 'activity' for each 'case_id', keeping the first occurrence
        first_occurrences = sorted_data.drop_duplicates(subset=['case_id', 'activity'], keep='first')

        # Add a column that indicates the order of each activity within its case
        first_occurrences['activity_order'] = first_occurrences.groupby('case_id').cumcount() + 1

        # Pivot the table to have 'case_id' as index and 'activity' as columns,
        # values being the 'activity_order'
        pivot_table = first_occurrences.pivot(index='case_id', columns='activity', values='activity_order')
        
        pivot_table = pivot_table.fillna(0)
        pivot_table['score_seqential'] = pivot_table.apply(lambda row: self.score_sequential_calculation(row), axis=1)
        #apply score to self.data on case_id
        self.data['score_sequential'] = self.data['case_id'].apply(lambda x: pivot_table.loc[x]['score_seqential'])

    @exception_handler
    def calculate_found_scores(self):
        print(f"Calculating scores for {self.cat}")
        self.pivot_data_frequence()
        self.pivot_data_timestamps()

def main(MAC=False, DataName="sample"):
    data_handler = DataHandler(MAC=MAC, DataName=DataName)
    data, weights, log = data_handler.load_data()
    score_calculator = ProcessScoreCalculator(data, weights, log)
    return score_calculator

if __name__ == '__main__':
    result = main(MAC=True, DataName="sample")
    print(result.data)  # Optionally print or log the result for debugging
