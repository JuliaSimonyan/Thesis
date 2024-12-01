from sklearn.cluster import KMeans
import numpy as np

def perform_clustering(numeric_data, original_feature_data):
    """
    Perform clustering on the numeric data and add the cluster label to each feature.
    """
    # Convert the numeric data to a NumPy array
    data = np.array(numeric_data)

    # Perform KMeans clustering with 2 clusters (since you have 2 samples)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(data)

    # Add cluster labels to the original feature data
    for i, feature in enumerate(original_feature_data):
        feature['Cluster'] = clusters[i]  # Assign cluster label to each feature dictionary

    return original_feature_data
