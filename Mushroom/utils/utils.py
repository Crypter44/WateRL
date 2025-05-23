import json
import logging
import os
import random
import sys
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
        save_whole_file=False
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
        if save_whole_file:
            end = len(code)
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
                data[p1][p2][seed] = train(
                    p1, p2,
                    seed,
                    path,
                    last=(p1 == tuning_params1[-1] and p2 == tuning_params2[-1] and seed == seeds[-1])
                )
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
        j += gamma ** episode_steps * dataset[i]["rewards"][0]
        episode_steps += 1
        if dataset[i]["last"] or i == len(dataset) - 1:
            js.append(j)
            j = 0.
            episode_steps = 0

    if len(js) == 0:
        return [0.]
    return js


def exponential_reward(x, target, smoothness, bound, value_at_bound):
    b = np.log(value_at_bound) / (np.sqrt(smoothness) - np.sqrt(smoothness + bound ** 2))
    a = np.exp(b * np.sqrt(smoothness))
    return a * np.exp(-b * np.sqrt((x - target) ** 2 + smoothness))


def linear_reward(x, min_x, max_x, min_reward=0, max_reward=1.0):
    r = (max_x - x) / (max_x - min_x)
    r = np.clip(r, 0, 1)
    return min_reward + r * (max_reward - min_reward)


def final_evaluation(n_episodes_final, n_episodes_final_render, core, save_path):
    for i in range(n_episodes_final_render):
        core.evaluate(n_episodes=1, quiet=True)
        core.mdp.render(save_path=save_path + f"Final_{i}")

    for i, a in enumerate(core.agents):
        a.save(save_path + f"/checkpoints/Final_Agent_{i}")

    data = {}
    if n_episodes_final > 0:
        with open(save_path + "Evaluation.json", "w") as f:
            final = compute_metrics_with_labeled_dataset(
                core.evaluate(n_episodes=n_episodes_final, render=False)[0],
            )
            data = {
                "Min": final[0],
                "Max": final[1],
                "Mean": final[2],
                "Median": final[3],
                "Count": final[4],
            }
            json.dump(data, f, indent=4)

    return data
