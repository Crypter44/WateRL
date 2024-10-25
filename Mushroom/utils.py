import os
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


def plot_J(Js):
    Js = np.array(Js)

    # Plot the cumulated reward
    plt.plot(np.mean(Js, axis=1), label="Mean J")
    plt.fill_between(np.arange(len(Js)), np.min(Js, axis=1), np.max(Js, axis=1), alpha=0.2, label="Range of J")
    plt.title("Cumulated reward per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("J")
    plt.legend()
    plt.show()