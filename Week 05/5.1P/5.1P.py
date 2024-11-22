import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv('data_correlation.csv')

# Function to calculate Pearson's correlation coefficient and print it along with the corrcoef matrix
def print_correlation(x, y, label1, label2):
    pearson_r = np.corrcoef(x, y)[0, 1]
    corrcoef_matrix = np.corrcoef(x, y)
    print(f"{label1} and {label2} pearson_r: {pearson_r}")
    print(f"{label1} and {label2} corrcoef: {corrcoef_matrix}\n")

# Calculate and print Pearson's correlation coefficients and corrcoef matrices
print_correlation(df['a'], df['b'], 'a', 'b')
print_correlation(df['a'], df['c'], 'a', 'c')
print_correlation(df['a'], df['d'], 'a', 'd')

# Function to plot scatter plot with optional line of best fit
def plot_correlation(x, y, title, ax):
    ax.scatter(x, y,)
    ax.set_xlabel('a')
    ax.set_ylabel(title)
    ax.legend()
    
    if abs(np.corrcoef(x, y)[0, 1]) > 0.5:
        # Fit line
        line_coef = np.polyfit(x, y, 1)
        xx = np.arange(min(x), max(x), 0.1)
        yy = line_coef[0] * xx + line_coef[1]
        ax.plot(xx, yy, 'r-', lw=2)

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(7, 15), dpi=100)

# Plot correlations
plot_correlation(df['a'], df['b'], 'b', axs[0])
plot_correlation(df['a'], df['c'], 'c', axs[1])
plot_correlation(df['a'], df['d'], 'd', axs[2])

plt.tight_layout()
plt.show()