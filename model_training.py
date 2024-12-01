import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from feature_extraction import extract_features
from snippets import bubble_sort, quick_sort
from clustering import perform_clustering
from optimization import memoize, loop_unrolling

# Function to measure execution time of a function
def measure_execution_time(func, arr):
    start_time = time.time()
    func(arr)
    return time.time() - start_time

# Apply all optimizations and return execution time improvements for each method
def apply_all_optimizations(func, arr):
    """
    Apply all optimization methods and return execution time improvements for each method.
    """
    # Apply optimizations to the function `func` with the array `arr`
    optimizations = {
        'memoize': memoize(func, arr),  # Pass both `func` and `arr`
        'loop_unrolling': loop_unrolling(arr),  # Pass `arr` to `loop_unrolling`
        # Add other optimizations as needed
    }
    
    improvements = {}
    for opt, optimized_func in optimizations.items():
        execution_time_improvement = measure_execution_time(optimized_func, arr)
        improvements[f"{opt}_Execution Time Improvement"] = execution_time_improvement
    
    return improvements

# Function to train the model
def train_model(df):
    features = df.drop(columns=['Function Name'])
    target = df[['memoize_Execution Time Improvement', 'loop_unrolling_Execution Time Improvement']]  # Example target columns

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    return model

# Function to predict best optimizations based on cluster features
def predict_best_optimizations(model, cluster_features):
    """
    Predict the best optimization methods for a cluster using the trained model.
    """
    predicted_improvements = model.predict([cluster_features])  # Get improvements for all methods
    best_optimizations = []

    # Apply a threshold or logic to decide the best optimizations
    for i, improvement in enumerate(predicted_improvements[0]):
        if improvement > 0:  # Example: choosing optimizations with positive improvements
            best_optimizations.append(improvement)
    
    return best_optimizations

# Function to collect data, apply clustering and optimizations
def collect_data():
    """
    Collect feature data, apply clustering, and collect optimization results.
    """
    # Define the functions directly
    functions = [bubble_sort, quick_sort]  # These should be function objects, not strings.
    arr = [5, 2, 9, 1, 5, 6]  # Example array for sorting optimization
    
    feature_data = []
    
    # Extract features from functions
    for func in functions:
        feature = extract_features(func, arr)  # Example input data
        feature_data.append(feature)

    # Prepare the data for clustering (extract numeric features)
    numeric_data = []
    for feature in feature_data:
        # Extract only the numeric values: Execution Time and Memory Usage
        feature_values = [feature['Execution Time'], feature['Memory Usage (in MiB)']]
        numeric_data.append(feature_values)

    # Perform clustering with numeric data
    clustered_data = perform_clustering(numeric_data, feature_data)  # Pass both numeric and original feature data

    # Collect optimization results for each function
    df_rows = []
    for feature in clustered_data:
        func = feature['Function Name']  # This should be a function object, not a string
        
        # Pass the function and array to apply_all_optimizations
        optimization_results = apply_all_optimizations(func, arr)  # Pass `func` and `arr` (the array)
        
        row = {**feature, **optimization_results}
        df_rows.append(row)

    # Create DataFrame from the collected data
    df = pd.DataFrame(df_rows)
    return df

# Main execution if this module is run directly
if __name__ == '__main__':
    df = collect_data()  # Collect feature and optimization data

    # Train the model using the collected data
    model = train_model(df)

    # Example cluster features (this should be dynamically set based on the actual cluster data)
    cluster_features = [0.002, 0.01, 100, 0.5]  # Example feature vector for a cluster

    # Predict the best optimizations for the given cluster
    best_optimizations = predict_best_optimizations(model, cluster_features)
    print(f"Best optimizations for this cluster: {best_optimizations}")

    # For each optimization method, apply it and show before and after performance
    for opt in best_optimizations:
        optimized_result = apply_all_optimizations(bubble_sort, [5, 2, 9, 1, 5, 6])[f'{opt}_Execution Time Improvement']
        print(f"Optimization: {opt}, Improvement: {optimized_result}")
