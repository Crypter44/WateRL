import os
import random
from typing import Callable

import numpy as np
import torch
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm

from Mushroom.plotting import _plot_metrics_to_ax


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
        _plot_metrics_to_ax(ax, data, title, range_alpha)
    else:
        for i, (seed, metrics) in enumerate(data.items()):
            # Get the default color cycle
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            _plot_metrics_to_ax(ax[i], {seed: metrics}, title, range_alpha, color_cycle[i])

    return fig, ax


def plot(tuning_params1, tuning_params2, seeds, data):
    # Plot the results
    fig, ax = plt.subplots(
        len(tuning_params1),
        len(tuning_params2),
        figsize=(len(tuning_params2) * 8, len(tuning_params1) * 8)
    )

    x = 0
    for p1 in tuning_params1:
        y = 0
        for p2 in tuning_params2:
            if len(tuning_params1) == 1 and len(tuning_params2) == 1:
                _plot_metrics_to_ax(ax, data[f"{p1}-{p2}"], f"p1={p1} p2={p2}", 0.1)
            elif len(tuning_params1) == 1:
                _plot_metrics_to_ax(ax[y], data[f"{p1}-{p2}"], f"p1={p1} p2={p2}", 0.1)
            elif len(tuning_params2) == 1:
                _plot_metrics_to_ax(ax[x], data[f"{p1}-{p2}"], f"p1={p1} p2={p2}", 0.1)
            else:
                _plot_metrics_to_ax(ax[x][y], data[f"{p1}-{p2}"], f"p1={p1} p2={p2}", 0.1)
            y += 1
        x += 1

    fig.show()


def grid_search(tuning_params1, tuning_params2, seeds, train: Callable, base_path):
    data = {}
    base_path += datetime.now().strftime("%y-%m-%d:%H-%M/")
    experiment_bar = tqdm(total=len(tuning_params1) * len(tuning_params2), unit='experiment')
    for p1 in tuning_params1:
        data[p1] = {}
        for p2 in tuning_params2:
            data[p1][p2] = {}
            seed_bar = tqdm(seeds, unit='seed', leave=False)
            for seed in seed_bar:
                set_seed(seed)
                path = base_path + f"{p1}-{p2}/s{seed}/"
                os.makedirs(path, exist_ok=True)
                data[p1][p2][seed] = train(p1, p2, seed, path)
            experiment_bar.update()
    return data, base_path
