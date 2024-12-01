from snippets import bubble_sort, quick_sort  # Import your code snippets
from feature_extraction import extract_features
from clustering import perform_clustering

if __name__ == "__main__":  # Ensure safe multiprocessing
    # List of functions to evaluate (you can add more here)
    functions = [bubble_sort, quick_sort]

    # Extract features for each function
    features = []
    for func in functions:
        features.append(extract_features(func, [5, 2, 9, 1, 5, 6]))  # Example input array

    # Print out the extracted features
    print("Extracted Features:")
    for feature in features:
        print(feature)

    # Perform clustering on the extracted features
    features_with_clusters = perform_clustering(features)

    # Print out the features with their cluster assignments
    print("Features with Clusters:")
    for feature in features_with_clusters:
        print(feature)

        # from snippets import *
# from feature_extraction import extract_features
# from clustering import perform_clustering
# from optimization import memoize, loop_unrolling
# from evaluation import evaluate_performance

# # List of functions and their corresponding arguments
# snippets = [
#     (bubble_sort, [[64, 34, 25, 12, 22, 11, 90]]),
#     (merge_sort, [[64, 34, 25, 12, 22, 11, 90]]),
#     (quick_sort, [[64, 34, 25, 12, 22, 11, 90]]),
#     (insertion_sort, [[64, 34, 25, 12, 22, 11, 90]]),
#     (selection_sort, [[64, 34, 25, 12, 22, 11, 90]]),
#     (heap_sort, [[64, 34, 25, 12, 22, 11, 90]]),
#     (binary_search, [[1, 2, 3, 4, 5, 6, 7, 8, 9], 4]),
#     (linear_search, [[64, 34, 25, 12, 22, 11, 90], 22]),
#     (factorial, [10]),
#     (fibonacci, [10]),
#     (power, [2, 10]),
#     (gcd, [48, 18]),
#     (lcm, [48, 18]),
#     (matrix_multiply, [[[1, 2], [3, 4]], [[2, 0], [1, 2]]]),
#     (sieve_of_eratosthenes, [30]),
#     (tower_of_hanoi, [3, 'A', 'C', 'B']),
#     (knapsack, [50, [10, 20, 30], [60, 100, 120], 3]),
#     (dijkstra, [[[0, 9, 75, 0, 0, 0],
#                  [9, 0, 95, 19, 42, 0],
#                  [75, 95, 0, 51, 66, 45],
#                  [0, 19, 51, 0, 31, 0],
#                  [0, 42, 66, 31, 0, 29],
#                  [0, 0, 45, 0, 29, 0]], 0, 6]),
#     (dfs, [{0: [1, 2], 1: [0, 3, 4], 2: [0, 4], 3: [1], 4: [2, 3]}, 0]),
#     (bfs, [{0: [1, 2], 1: [0, 3, 4], 2: [0, 4], 3: [1], 4: [2, 3]}, 0]),
# ]

# # Extract features for each snippet
# features = [extract_features(func, *args) for func, args in snippets]

# # Perform clustering
# clusters = perform_clustering(features)

# # General optimization application
# optimized_times = []
# initial_times = []

# for func, args in snippets:
#     # Measure initial performance
#     initial_features = extract_features(func, *args)
#     initial_times.append(initial_features['Execution Time'])

#     # Apply optimization based on function type
#     if func.__name__ in ['fibonacci', 'factorial', 'tower_of_hanoi']:
#         optimized_func = memoize(func)
#         optimized_features = extract_features(optimized_func, *args)
#     elif func.__name__ in ['bubble_sort', 'insertion_sort']:
#         # Apply loop unrolling before calling the sorting function
#         arr = args[0].copy()  # Create a copy of the array
#         loop_unrolling(arr)
#         optimized_func = func  # Apply the original sorting function
#         optimized_features = extract_features(optimized_func, arr)
#     else:
#         optimized_features = extract_features(func, *args)

#     optimized_times.append(optimized_features['Execution Time'])
