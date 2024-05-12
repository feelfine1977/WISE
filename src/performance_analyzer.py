import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


from process_score_calculator import prepare_calculations as calculate_scores

class PerformanceAnalyzer:
    def __init__(self, data, index='case_Purchasing_Document', threshold=0.10, DataName="BPIC_2019", json_mapping=None):
        self.data = data.fillna(0)
        self.threshold = threshold
        self.index = index
        self.DataName = DataName
        self.json_mapping = json_mapping or {}
        self.score_columns, self.category_columns = self.get_columns(data)

    def get_columns(self, data):
        score_columns = [col for col in data.columns if col.startswith('score_')]
        category_columns = [col for col in data.columns if col.startswith('cat_dim')]
        return score_columns, category_columns

    def calculate_outliers(self, data, method='standard'):
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        if method == 'adjusted':
            y = np.asarray(data, dtype=np.double)
            mc = sm.stats.stattools.medcouple(y)
            exp_factor = 3 if mc >= 0 else 4
            lower_bound = q1 - 1.5 * iqr * np.exp(-exp_factor * mc)
            upper_bound = q3 + 1.5 * iqr * np.exp(exp_factor * mc)
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
        whis = [lower_bound, upper_bound]
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return lower_bound, upper_bound, whis, outliers

    def plot_custom_boxplot(self, score_column_name, category_column, method='standard', title=None):
        plt.figure(figsize=(12, 8))
        categories = self.data[category_column].unique()
        colors = sns.color_palette("RdYlGn", len(categories))
        
        for i, category in enumerate(categories):
            category_data = self.data[self.data[category_column] == category][score_column_name].dropna()
            if not category_data.empty:
                lower_bound, upper_bound, _, _ = self.calculate_outliers(category_data, method=method)
                
                # Convert bounds to percentiles for whisker positions
                # Finding percentiles that correspond to the actual data values of lower_bound and upper_bound
                lower_whisker = np.percentile(category_data, np.interp(lower_bound, np.sort(category_data), np.linspace(0, 100, len(category_data))))
                upper_whisker = np.percentile(category_data, np.interp(upper_bound, np.sort(category_data), np.linspace(0, 100, len(category_data))))
                
                sns.boxplot(x=category_column, y=score_column_name, data=self.data[self.data[category_column] == category],
                            whis=[lower_whisker, upper_whisker], color=colors[i])

        plt.xticks(rotation=45)
        plt.title(title if title else f'Custom Box Plot of {score_column_name} by {category_column}', fontsize=16)
        plt.xlabel(category_column, fontsize=14)
        plt.ylabel(score_column_name, fontsize=14)
        sns.despine(trim=True)
        plt.tight_layout()
        plt.show()

    def analyze_performance(self, visualize=False, category_vis=None, score_vis=None, method='standard'):
        if visualize:
            self.plot_custom_boxplot(score_vis, category_vis, method=method, title='Adjusted Box Plot of Scores by Category')

def perform_analysis(visualize=False, category=None, score=None, DataName="BPIC_2019", method='standard', data_filtered=None):
    if data_filtered is not None:
        data = data_filtered
    else:
        data = calculate_scores(MAC=True, DataName=DataName, layer="General_Process_Standards").data
    json_mapping = {}  # Assuming a dictionary is set up here
    analyzer = PerformanceAnalyzer(data, DataName=DataName, json_mapping=json_mapping)
    analyzer.analyze_performance(visualize=visualize, category_vis=category, score_vis=score, method=method)

if __name__ == '__main__':
    perform_analysis(visualize=True, category='cat_dim_6', score='mean_score', method='standard')
