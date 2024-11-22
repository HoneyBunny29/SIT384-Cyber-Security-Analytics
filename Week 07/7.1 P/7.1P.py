import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Set seed for reproducibility
np.random.seed(0)

# Create synthetic dataset
X, _ = make_blobs(n_samples=200, centers=[[3, 2], [6, 4], [10, 5]], cluster_std=0.9)

# Plot the created dataset
plt.figure(figsize=(8, 4))
plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', s=30)
plt.title("Created Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# K-Means clustering
kmeans = KMeans(init="k-means++", n_clusters=3, n_init=12)
kmeans.fit(X)
centroids = kmeans.cluster_centers_

plt.figure(figsize=(8, 4), facecolor='black')
ax = plt.gca()
ax.set_facecolor('black')

colors = ['white', 'white', 'white']  # All points in white
for i in range(3):
    cluster_points = X[kmeans.labels_ == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], marker='o', s=30)
    plt.plot(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], linestyle='-', linewidth=1)

# Plot centroids with specified colors
centroid_colors = ['yellow', 'blue', 'red']
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c=centroid_colors)

# Remove title, axis labels, and legend
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.legend().set_visible(False)  # Hide legend
plt.grid(False)  # Disable the grid
plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.show()

# Agglomerative Hierarchical clustering
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='average')
agg_clustering.fit(X)

# Define markers for clusters
markers = ['o', 's', '^']  # Circle, square, and triangle for clusters 0, 1, 2
colors = ['blue', 'green', 'purple']

# Plot the Agglomerative Hierarchical clusters with different markers
plt.figure(figsize=(8, 4))
for i in range(3):
    cluster_points = X[agg_clustering.labels_ == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], marker=markers[i], s=100, label=f'Cluster {i}')

plt.title("Agglomerative Hierarchical")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.legend()
plt.show()

# Calculate distance matrix and plot dendrogram
linked = linkage(X, 'complete')

plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Dendrogram (Complete Linkage)")
plt.show()