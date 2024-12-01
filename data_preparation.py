# data_preparation.py
import pandas as pd
from performance_measurement import measure_improvement
from optimization import memoize, loop_unrolling
from feature_extraction import extract_features
from snippets import *

def prepare_data(funcs, optimizations, *args):
    data = []
    for func in funcs:
        # Get original feature
        original_features = extract_features(func, *args)
        row = {
            'Function Name': func.__name__,
            'Original Execution Time': original_features['Execution Time'],
            'Original Memory Usage': original_features['Memory Usage (in MiB)'],
        }
        
        # Measure performance after each optimization
        for optimization in optimizations:
            improvement = measure_improvement(func, optimization, *args)
            row[f'{optimization.__name__}_Execution Time Improvement'] = improvement['Execution Time Improvement']
            row[f'{optimization.__name__}_Memory Usage Improvement'] = improvement['Memory Usage Improvement']
        
        data.append(row)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    return df

# Example usage:
funcs = [bubble_sort, quick_sort]  # List of functions
optimizations = [memoize, loop_unrolling]  # List of optimizations
args = ([5, 2, 9, 1, 5, 6],)  # Example input

df = prepare_data(funcs, optimizations, *args)
print(df)
