import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pdpbox import pdp, info_plots
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

from process_score_calculator import prepare_calculations as calculate_scores

class PerformanceAnalyzer:
    def __init__(self, data, index='case_Purchasing_Document', threshold=0.10, DataName="BPIC_2019", json_mapping=None):
        self.data = data.fillna(0)
        self.threshold = threshold
        self.index = index
        self.DataName = DataName
        self.json_mapping = json_mapping or {}
        self.score_columns, self.category_columns = self.get_columns(data)
        self.save_path = '/Users/urszulajessen/code/gitHub/WISE/data/results/data_BPIC_2019/'

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

    def normalize_scores(self):
        # Normalize the score columns between 0 and 1
        for col in self.score_columns:
            min_val = self.data[col].min()
            max_val = self.data[col].max()
            self.data[col] = (self.data[col] - min_val) / (max_val - min_val)
        # Recalculate the mean_score based on normalized values
        self.data['mean_score'] = self.data[self.score_columns].mean(axis=1)
        # Ensure mean_score is included in the score columns
        if 'mean_score' not in self.score_columns:
            self.score_columns.append('mean_score')

    def plot_custom_boxplot(self, score_column_name, category_column, method='standard', title=None, save_path=None):
        plt.figure(figsize=(12, 8))
        categories = self.data[category_column].unique()
        colors = sns.color_palette("deep", len(categories))
        
        for i, category in enumerate(categories):
            category_data = self.data[self.data[category_column] == category][score_column_name].dropna()
            if not category_data.empty:
                lower_bound, upper_bound, _, _ = self.calculate_outliers(category_data, method=method)
                
                lower_whisker = np.percentile(category_data, np.interp(lower_bound, np.sort(category_data), np.linspace(0, 100, len(category_data))))
                upper_whisker = np.percentile(category_data, np.interp(upper_bound, np.sort(category_data), np.linspace(0, 100, len(category_data))))
                
                sns.boxplot(x=category_column, y=score_column_name, data=self.data[self.data[category_column] == category],
                            whis=[lower_whisker, upper_whisker], color=colors[i])
        plt.xticks(rotation=45 if len(categories) <= 10 else 90)
        plt.title(title if title else f'Custom Box Plot of {score_column_name} by {category_column}', fontsize=16)
        plt.xlabel(category_column, fontsize=14)
        plt.ylabel(score_column_name, fontsize=14)
        sns.despine(trim=True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def save_custom_boxplot(self, score_column_name, category_column, method='standard', title=None, save_path=None):
        self.plot_custom_boxplot(score_column_name, category_column, method=method, title=title, save_path=save_path)

    def create_combined_heatmap(self, category_column, save_path=None):
        pivot_table = self.data.pivot_table(index=category_column, values=self.score_columns, aggfunc='mean')
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", cbar=True)
        plt.title(f'Heatmap of Scores by {category_column}', fontsize=16)
        plt.xlabel('Scores', fontsize=14)
        plt.ylabel(category_column, fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def create_sensitivity_analysis_heatmap(self, save_path=None):
        variability = {cat_column: self.data.groupby(cat_column)[self.score_columns].std().mean().mean()
                       for cat_column in self.category_columns}
        worst_cat_dim = max(variability, key=variability.get)
        
        worst_pivot_table = self.data.pivot_table(index=worst_cat_dim, values=self.score_columns, aggfunc='mean')
        plt.figure(figsize=(14, 10))
        sns.heatmap(worst_pivot_table, annot=True, cmap="YlGnBu", cbar=True)
        plt.title(f'The category {worst_cat_dim} should be analysed further as it displays the highest variability across different scores', fontsize=16)
        plt.xlabel('Scores', fontsize=14)
        plt.ylabel(worst_cat_dim, fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        return worst_cat_dim

    def create_heatmap(self, category, save_path=None):
        self.create_combined_heatmap(category, save_path=save_path)


    def plot_all_heatmaps(self):
        for category_column in self.category_columns:
            self.create_heatmap(category_column, save_path=f'{self.save_path}{category_column}_heatmap.png')

    def identify_lowest_scores(self, n=5):
        lowest_scores = self.data.nsmallest(n, 'mean_score')
        print("Lowest scores in the dataset:")
        print(lowest_scores)

    def analyze_performance(self, visualize=False, category_vis=None, score_vis=None, method='standard', sensitivity_analysis=False, all_plots=False):
        self.normalize_scores()
        if visualize:
            if all_plots:
                self.plot_all_heatmaps()
                self.plot_custom_boxplot(score_vis, category_vis, method=method, title='Adjusted Box Plot of Scores by Category', save_path=f'{self.save_path}{category_vis}_adjusted_boxplot.png')

            else:
                self.plot_custom_boxplot(score_vis, category_vis, method=method, title='Adjusted Box Plot of Scores by Category', save_path=f'{self.save_path}{category_vis}_adjusted_boxplot.png')
                self.create_combined_heatmap(category_vis, save_path=f'{self.save_path}{category_vis}_heatmap.png')
        if sensitivity_analysis:
            self.create_sensitivity_analysis_heatmap(save_path=f'{self.save_path}sensitivity_analysis_heatmap.png')

def perform_analysis(visualize=False, category=None, score=None, DataName="BPIC_2019", method='standard', data_filtered=None, all_plots=False):
    if data_filtered is not None:
        data = data_filtered
    else:
        data = calculate_scores(MAC=True, DataName=DataName, layer="General_Process_Standards").data
    json_mapping = {}
    analyzer = PerformanceAnalyzer(data, DataName=DataName, json_mapping=json_mapping)
    analyzer.analyze_performance(visualize=visualize, category_vis=category, score_vis=score, method=method, all_plots=all_plots)

if __name__ == '__main__':
    perform_analysis(visualize=True, category='cat_dim_5', score='mean_score', method='adjusted', DataName="sample", all_plots=True)
