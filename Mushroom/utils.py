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
    rows = 1 if multiple_seeds_per_plot else len(data)
    fig, ax = plt.subplots(rows, 1, figsize=(8 if multiple_seeds_per_plot else rows * 3, 8))

    if multiple_seeds_per_plot or len(data) == 1:
        plot_to_ax(ax, data, title, range_alpha)
    else:
        for i, (seed, metrics) in enumerate(data.items()):
            # Get the default color cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            plot_to_ax(ax[i], {seed: metrics}, title, range_alpha, color_cycle[i])

    return fig, ax


def plot_to_ax(ax, data: dict, title: str, range_alpha=0.1, color=None):
    for seed, metrics in data.items():
        ax.plot(metrics[:, 2], label=f"Mean score, s={seed}", color=color)

    for seed, metrics in data.items():
        if color is not None:
            ax.fill_between(np.arange(len(metrics)), metrics[:, 0], metrics[:, 1], alpha=range_alpha,
                            label=f"Range, s={seed}", color=color)
        else:
            ax.fill_between(np.arange(len(metrics)), metrics[:, 0], metrics[:, 1], alpha=range_alpha,
                            label=f"Range, s={seed}")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()


def plot_data(tuning_params1, tuning_params2, seeds, data):
    # Plot the results
    fig, ax = plt.subplots(
        len(tuning_params1),
        len(tuning_params2),
        figsize=(len(tuning_params2) * 8, len(tuning_params1) * 8)
    )

    x = 0
    y = 0
    for p1 in tuning_params1:
        y = 0
        for p2 in tuning_params2:
            fig1, ax1 = plot_multiple_seeds(data[f"{p1}-{p2}"], f"p1={p1} p2={p2}", True)
            fig1.show()
            if len(seeds) > 1:
                fig2, ax2 = plot_multiple_seeds(data[f"{p1}-{p2}"], f"p1={p1} p2={p2}", False)
                fig2.show()

            if len(tuning_params1) == 1 and len(tuning_params2) == 1:
                plot_to_ax(ax, data[f"{p1}-{p2}"], f"p1={p1} p2={p2}", 0.1)
            elif len(tuning_params1) == 1:
                plot_to_ax(ax[y], data[f"{p1}-{p2}"], f"p1={p1} p2={p2}", 0.1)
            elif len(tuning_params2) == 1:
                plot_to_ax(ax[x], data[f"{p1}-{p2}"], f"p1={p1} p2={p2}", 0.1)
            else:
                plot_to_ax(ax[x][y], data[f"{p1}-{p2}"], f"p1={p1} p2={p2}", 0.1)
            y += 1
        x += 1

    fig.show()
