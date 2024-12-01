import pandas as pd
import matplotlib.pyplot as plt
from model_training import train_model, collect_data, apply_all_optimizations, predict_best_optimizations
from feature_extraction import extract_features
from snippets import bubble_sort, quick_sort
from clustering import perform_clustering

if __name__ == '__main__':

    # Step 1: Collect data and train the model
    df = collect_data()  # Collects features and optimization results
    model = train_model(df)  # Trains a model to predict the best optimization methods

    # Check if the model is initialized
    if model is None:
        print("Model training failed!")
        exit()  # Or handle the error appropriately

    # Example cluster features
    cluster_features = [0.002, 0.01, 100, 0.5]  # Example feature vector (Execution Time, Memory Usage, etc.)

    # Step 2: Predict the best optimizations for a cluster
    best_optimizations = predict_best_optimizations(model, cluster_features)
    print(f"Best optimizations for this cluster: {best_optimizations}")

    # Step 3: Visualize clusters and optimizations
    functions = [bubble_sort, quick_sort]  # Add all your code snippets here

    # Collect features and perform clustering
    feature_data = []
    for func in functions:
        feature = extract_features(func, [5, 2, 9, 1, 5, 6])  # Example input
        feature_data.append(feature)

    clustered_data = perform_clustering(feature_data)

    # Get cluster assignments
    clusters = [feature['Cluster'] for feature in clustered_data]
    execution_times = [feature['Execution Time'] for feature in clustered_data]
    memory_usages = [feature['Memory Usage (in MiB)'] for feature in clustered_data]

    # Step 4: Visualize clusters using a scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(execution_times, memory_usages, c=clusters, cmap='viridis', s=100, alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Execution Time (s)')
    plt.ylabel('Memory Usage (MiB)')
    plt.title('Clusters of Code Snippets')
    plt.show()

    # Step 5: Show the best optimizations for each cluster
    for cluster in set(clusters):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster]
        print(f"\nCluster {cluster}:")
    
        for idx in cluster_indices:
            # Get best optimizations for this cluster
            optimizations_for_cluster = best_optimizations  # This should be updated dynamically per cluster
            print(f"Code {idx} - Best Optimizations: {optimizations_for_cluster}")
            
            # Before and after applying optimizations
            before_optimization = df.iloc[idx]['Execution Time']
            after_optimization = {opt: apply_all_optimizations(bubble_sort, [5, 2, 9, 1, 5, 6])[f'{opt}_Execution Time Improvement'] for opt in best_optimizations}
            
            print(f"Before Optimization: {before_optimization}, After Optimization: {after_optimization}")
