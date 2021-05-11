import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_locations(locations,):
    """
    Displays "Location intensity" on the image grid of each item
    """
    palette = sns.color_palette('muted')
    for idx, item in enumerate(locations):
        coordinates = np.asarray(locations[item])
        plot = sns.jointplot(coordinates[:,0], coordinates[:,1], ylim=160, xlim=[0, 160], color=palette[0])
        plot.plot_joint(sns.kdeplot, color=palette[1], zorder=0, levels=10, linewidth=10)
        plot.fig.suptitle(f'Distribution of {item} positions')
        plot.fig.subplots_adjust(top=0.95)


def plot_dist_distributions(distances, figsize=(20, 20)):
    """
    Plots paired eucledian distance distributions for each pair of objects
    """
    ncols = int(len(distances)**0.5)
    fig, axs = plt.subplots(
        constrained_layout=True,
        nrows=6,
        ncols=5,
        figsize=figsize,
        sharey='all'
        )
    fig.suptitle('Eucledian distance distributions for each pair of objects', fontsize='x-large', fontweight='bold')
    row = 0
    count = 1
    col = 0
    palette = sns.color_palette('pastel')
    for idx, pair in enumerate(distances):
        if count % 5 == 0:
            axs[row, col] = sns.histplot(
                distances[pair], 
                stat='density', 
                element='step', 
                kde=True, 
                ax=axs[row, col],
                color=palette[row]
            )
            axs[row, col].set_title(f'Distribution for\n{pair}', {'fontweight':'medium'})
            axs[row, col].set_xlabel('Distance, (pixels)')
            row += 1
            count += 1
            col = 0
        else:
            axs[row, col] = sns.histplot(
                distances[pair], 
                stat='density', 
                element='step', 
                kde=True, 
                ax=axs[row, col],
                color=palette[row]
            )
            axs[row, col].set_title(f'Distribution for\n{pair}', {'fontweight':'medium'})
            axs[row, col].set_xlabel('Distance, (pixels)')
            count += 1
            col += 1