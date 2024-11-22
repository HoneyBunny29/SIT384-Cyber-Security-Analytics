import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
file_name = 'Malicious_or_criminal_attacks_breakdown-Top_five_sectors_July-Dec-2023.csv'
df = pd.read_csv('Malicious_or_criminal_attacks_breakdown-Top_five_sectors_July-Dec-2023.csv', index_col=0, engine='python')

# Data for plotting
sectors = df.columns
attack_types = df.index
colors = ['red', 'yellow', 'blue', 'green']

# Create subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 9), dpi=100)

# Grouped Bar Chart
width = 0.2  # Width of the bars
x = range(len(sectors))  # X locations for the groups

for i, attack_type in enumerate(attack_types):
    ax1.bar([p + width*i for p in x], df.loc[attack_type], width, label=attack_type, color=colors[i])
    
ax1.set_xlabel('Top five industry sectors')
ax1.set_ylabel('Number of attacks')
ax1.set_title('Malicious or criminal attack breaches – Top 5 sectors')
ax1.set_xticks([p + 1.5 * width for p in x])
ax1.set_xticklabels(sectors, rotation=90)
ax1.legend(title='Attack types')

# Show values on top of bars
for bars in ax1.containers:
    ax1.bar_label(bars)

# Stacked Bar Chart
bottom = [0] * len(sectors)
for i, attack_type in enumerate(attack_types):
    ax2.bar(sectors, df.loc[attack_type], bottom=bottom, label=attack_type, color=colors[i])
    bottom = [i+j for i, j in zip(bottom, df.loc[attack_type])]

ax2.set_xlabel('Top five industry sectors')
ax2.set_ylabel('Number of attacks')
ax2.set_title('Malicious or criminal attack breaches – Top 5 sectors')
ax2.set_xticklabels(sectors, rotation=90)
ax2.legend(title='Attack types')

plt.tight_layout()
plt.show()