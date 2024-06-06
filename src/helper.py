import pandas as pd
from functools import wraps
import os 
import json

from abc import ABC, abstractmethod

def load_settings():
    # Get the absolute path of the settings.json file
    settings_path = os.path.join(os.path.dirname(__file__), '../settings.json')
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    return settings

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