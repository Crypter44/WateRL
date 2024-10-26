import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def set_seed(seed: int):
    """
    Set the seed of the random number generators of numpy, torch and random.
    :param seed: The seed to set.
    :return: None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def plot_multiple_seeds(data: dict, title: str, multiple_seeds_per_plot=True, range_alpha=0.1):
    fig, ax = plt.subplots(1 if multiple_seeds_per_plot else len(data), 1, figsize=(8, 8))

    if multiple_seeds_per_plot:
        plot_to_ax(ax, data, title, range_alpha)
    else:
        for i, (seed, metrics) in enumerate(data.items()):
            # Get the default color cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            print(color_cycle[i])

            plot_to_ax(ax[i], {seed: metrics}, title, range_alpha, color_cycle[i])


    return fig, ax

def plot_to_ax(ax, data: dict, title: str, range_alpha=0.1, color = None):
    for seed, metrics in data.items():
        ax.plot(metrics[:, 2], label=f"Mean score, s={seed}", color=color)

    for seed, metrics in data.items():
        if color is not None:
            ax.fill_between(np.arange(len(metrics)), metrics[:, 0], metrics[:, 1], alpha=range_alpha, label=f"Range, s={seed}", color=color)
        else:
            ax.fill_between(np.arange(len(metrics)), metrics[:, 0], metrics[:, 1], alpha=range_alpha, label=f"Range, s={seed}")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()

