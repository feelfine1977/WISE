import pandas as pd
from functools import wraps

from abc import ABC, abstractmethod

def exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred in {func.__name__}: {str(e)}")
            return None
    return wrapper

def get_category_data(categories,score_type, category_case):
        # Extract data for the specific category and score type
        for category in categories['Categories']:
            if category['Category'] == category_case:
                return category[score_type]
        return None


class ScoreCalculator(ABC):
    def __init__(self, data, weights, log, cat):
        self.data = data
        self.weights = weights
        self.log = log
        self.cat = cat
    
    @abstractmethod
    def calculate_score(self, row):
        pass