# model_inference.py
from data_preparation import *
def predict_best_optimizations(model, func, optimizations, *args):
    # Prepare input features
    features = prepare_data([func], optimizations, *args)
    
    # Get predictions
    predictions = model.predict(features)
    print(f'Predictions for {func.__name__}:')
    print(predictions)

# Example usage
predict_best_optimizations(model, bubble_sort, [memoize, loop_unrolling], [5, 2, 9, 1, 5, 6])
