import numpy as np

def perform_clustering(feature_data):
    # Convert to numpy array
    data = np.array(feature_data)
    
    # Check if the data contains any non-numeric entries and convert them to NaN
    # This step ensures that we don't run into type issues.
    data = np.where(np.isnan(data.astype(float)), np.nan, data)  # Convert to float if possible
    
    # Remove rows with NaN values
    data = data[~np.isnan(data).any(axis=1)]
    
    # Perform clustering here, e.g., KMeans or other clustering algorithms.
    # This is just an example; replace with your actual clustering code.
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    
    # Return clustering result (cluster assignments, features, etc.)
    return data
