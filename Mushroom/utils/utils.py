import os
import random
from datetime import datetime
from typing import Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from Mushroom.utils.plotting import _plot_metrics_to_ax


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


def parametrized_training(
        file_to_read_parameters_from,
        tuning_params1,
        tuning_params2,
        seeds,
        train: Callable,
        base_path,
):
    print("You are using parametrized training!\n"
          "This is a friendly reminder to make sure "
          "that the parameters are actually used (correctly) by the train function!")
    data = {}
    base_path += datetime.now().strftime("%y-%m-%d__%H:%M/")

    os.makedirs(base_path, exist_ok=True)
    with open(file_to_read_parameters_from, 'r') as f:
        code = f.read()
        # remove everything outside # PARAMS and # END_PARAMS
        begin, end = code.find("# PARAMS") + 8, code.find("# END_PARAMS")
        if begin == -1 or end == -1:
            raise ValueError("Parameters not found in file!")
        code = code[begin:end]
        with open(base_path + 'params.txt', 'w') as f2:
            f2.write(code)

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


def compute_metrics_with_labeled_dataset(dataset, gamma=1.):
    """
    Compute the metrics of each complete episode in the dataset.

    Args:
        dataset (list): the dataset to consider;
        gamma (float, 1.): the discount factor.

    Returns:
        The minimum score reached in an episode,
        the maximum score reached in an episode,
        the mean score reached,
        the median score reached,
        the number of completed episodes.

        If no episode has been completed, it returns 0 for all values.

    """
    for i in reversed(range(len(dataset))):
        if dataset[i]["last"]:
            i += 1
            break

    dataset = dataset[:i]

    if len(dataset) > 0:
        J = compute_J_with_labeled_dataset(dataset, gamma)
        return np.min(J), np.max(J), np.mean(J), np.median(J), len(J)
    else:
        return 0, 0, 0, 0, 0


def compute_J_with_labeled_dataset(dataset, gamma=1.):
    """
    Compute the cumulative discounted reward of each episode in the dataset.

    Args:
        dataset (list): the dataset to consider;
        gamma (float, 1.): discount factor.

    Returns:
        The cumulative discounted reward of each episode in the dataset.

    """
    js = list()

    j = 0.
    episode_steps = 0
    for i in range(len(dataset)):
        j += gamma ** episode_steps * dataset[i]["rewards"]
        episode_steps += 1
        if dataset[i]["last"] or i == len(dataset) - 1:
            js.append(j)
            j = 0.
            episode_steps = 0

    if len(js) == 0:
        return [0.]
    return js
