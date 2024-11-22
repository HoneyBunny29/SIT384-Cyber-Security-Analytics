import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D


# Load the dataset
cancer = load_breast_cancer()

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cancer.data)

# Apply PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Print the shapes and components
print("Original shape:", cancer.data.shape) 
print("Reduced shape:", X_pca.shape)         
print("PCA component shape:", pca.components_.shape)  
print("PCA components:")
print(pca.components_)

# Apply the Seaborn style
sns.set(style="darkgrid")

# Create a figure for the 2D plot
plt.figure(figsize=(10, 8))

# Plot points representing malignant cases 
plt.scatter(X_pca[cancer.target == 0, 0], X_pca[cancer.target == 0, 1], color='steelblue', marker='o', s=100, label='malignant', edgecolor='black')

# Plot points representing benign cases 
plt.scatter(X_pca[cancer.target == 1, 0], X_pca[cancer.target == 1, 1], color='coral', marker='^', s=100, label='benign', edgecolor='black')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')

# Add a legend 
plt.legend(loc='best')

# Display the plot
plt.show()

# Define a custom colormap
custom_cmap = LinearSegmentedColormap.from_list('swapped_coolwarm', ['#C21E56', 'slateblue'])

# 3D plot using the first 3 features of scaled data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
                     c=cancer.target, cmap=custom_cmap, label='malignant', s=12)

# Set title 
ax.set_title('First 3 features of scaled X')

# Adjust the viewing angle
ax.view_init(elev=7, azim=10)  

plt.legend()
plt.show()

# Define a custom colormap
custom_cmap = LinearSegmentedColormap.from_list('swapped_colors', ['#C21E56', 'slateblue'])

# 3D plot using the first two principal components after PCA
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], np.zeros_like(X_pca[:, 0]), 
                     c=cancer.target, cmap=custom_cmap, label='malignant', s=12)

# Set title 
ax.set_title('First two principal components of X after PCA transformation')

# Adjust the viewing angle
ax.view_init(elev=7, azim=10)  

plt.legend()
plt.show()