import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from process_score_calculator import prepare_calculations as calculate_scores

class PerformanceAnalyzer:
    def __init__(self, data, index='case_Purchasing_Document', threshold=0.10, DataName="BPIC_2019", json_mapping=None):
        self.data = data
        self.threshold = threshold
        self.index = index
        self.DataName = DataName
        self.json_mapping = json_mapping or {}
        self.score_columns, self.category_columns = self.get_score_and_category_columns(data)
        self.numeric_columns = [col for col in data.columns if col.startswith('num_dim')]
        self.original_category_names = self.map_category_names()  # Initialize mapping after fetching columns

    def map_category_names(self):
        # Map renamed category columns to their original names using json_mapping
        return {col: self.json_mapping.get(col, col) for col in self.category_columns}

    def get_score_and_category_columns(self, data):
        score_columns = [col for col in data.columns if col.startswith('score_')]
        category_columns = [col for col in data.columns if col.startswith('cat_dim')]
        return score_columns, category_columns

    def identify_poor_performers(self):
        self.data['mean_score'] = self.data[self.score_columns].mean(axis=1)
        lower_quantile_cutoff = self.data['mean_score'].quantile(self.threshold)
        poor_performers = self.data[self.data['mean_score'] <= lower_quantile_cutoff]
        if self.index in self.data.columns:
            poor_performers = poor_performers[[self.index, 'mean_score'] + self.score_columns].groupby(self.index).mean()
        return poor_performers

    def identify_category_worst_performers(self):
        results = {}
        for category in self.category_columns:
            original_name = self.original_category_names[category]
            mean_category_score = self.data.groupby(category)[self.score_columns].mean()
            worst_performer = mean_category_score.idxmin()
            worst_score = mean_category_score.min()
            name = f"{original_name}_worst_performer_general"
            results[original_name] = {name: {'worst_performer': worst_performer, 'score': worst_score}}
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv(f'{self.DataName}_category_worst_performers.csv')
        return results

    def extract_stats(self, category, score):
        self.data.fillna(0, inplace=True)
        self.data.dropna(subset=[category, score], inplace=True)
        
        grouped = self.data.groupby(category)[score]
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)
        iqr = q3 - q1
        median = grouped.median()
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = self.data.apply(lambda x: x[score] if (x[score] < lower_fence[x[category]] or x[score] > upper_fence[x[category]]) else None, axis=1)
        
        stats_df = pd.DataFrame({
            'Lower Whisker': lower_fence,
            'Q1': q1,
            'Median': median,
            'Q3': q3,
            'Upper Whisker': upper_fence,
            'Outliers': outliers.groupby(self.data[category]).apply(list)
        })
        return stats_df

    def save_stats_to_csv(self, file_prefix='boxplot_stats'):
        for category in self.category_columns:
            for score in self.score_columns:
                stats_df = self.extract_stats(category, score)
                stats_df.to_csv(f'{file_prefix}_{score}.csv')


    def run_analysis(self):
        poor_performers = self.identify_poor_performers()
        print("Poor Performers:", poor_performers)
        category_results = self.identify_category_worst_performers()
        print("Category Wise Worst Performers:")
        for category, details in category_results.items():
            print(f"Category: {category}")
            for key, val in details.items():
                print(f"{key}: Worst Performer: {val['worst_performer']}, Score: {val['score']}")
        #self.create_seaborn_boxplots(self.category_columns, self.score_columns)
        self.save_stats_to_csv()

# Main function to orchestrate the steps
def main():
    calculated_scores = calculate_scores(MAC=True, DataName="BPIC_2019", layer="General_Process_Standards")
    data = calculated_scores.data
    json_mapping = calculated_scores.json  # Assuming JSON is loaded or parsed here as a dictionary
    analyzer = PerformanceAnalyzer(data, index='case_Purchasing_Document', threshold=0.10, DataName="BPIC_2019", json_mapping=json_mapping)
    analyzer.run_analysis()

if __name__ == '__main__':
    main()
