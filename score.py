import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set the font globally
rcParams['font.family'] = 'Times New Roman'

def plot_intervals(axs, data, algorithm_name, intervals, row_offset):
    scores = data['scores']
    labels = data['labels']

    # Normalize the labels for coloring
    norm = plt.Normalize(labels.min(), labels.max())

    for idx, (start, end) in enumerate(intervals):
        interval_scores = scores[start:end]
        interval_labels = labels[start:end]
        row = (idx // 5) + row_offset
        col = idx % 5

        sc = axs[row, col].scatter(range(len(interval_scores)), interval_scores, c=interval_labels, cmap='coolwarm',
                                   norm=norm, s=10, alpha=0.7)
        axs[row, col].set_title(f'{algorithm_name} ({start} to {end})')
        axs[row, col].set_xlabel('Instance')
        axs[row, col].set_ylabel('Trustworthiness Score')

        # Add colorbar only to the last plot of each row
        if col == 4:
            fig.colorbar(sc, ax=axs[row, col], label='Label')


# Load the data for ADKGD
data_adkgd = np.load('top_930_scores_labels_adkgd.npz')

# Load the data for CAGED
data_caged = np.load('top_930_scores_labels_caged.npz')

# Define the intervals
intervals = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]

# Create a figure with 2x5 subplots
fig, axs = plt.subplots(2, 5, figsize=(20, 6))

# Plot intervals for ADKGD (first row)
plot_intervals(axs, data_adkgd, 'ADKGD', intervals, row_offset=0)

# Plot intervals for CAGED (second row)
plot_intervals(axs, data_caged, 'CAGED', intervals, row_offset=1)

# Adjust layout
plt.tight_layout()

# Save the plot as a PDF
plt.savefig('comparison_adkgd_caged_node_score_distribution.pdf', format='pdf', bbox_inches='tight')

# Display the plot
plt.show()
