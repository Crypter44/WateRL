import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def render_episode(core, fname="video.mp4", custom_tmp_path="./tmp"):
    """
    Render an episode as a video.

    For this to work, the environment must have a render method,
    that saves the current frame as a png file in the custom_tmp_path directory.
    :param core: The mushroom_rl Core object, containing the agent and environment.
    :param fname: The filename of the video.
    :param custom_tmp_path: The path to the temporary directory where the frames are stored.
    :return: None
    """
    try:
        if os.environ["RENDER"] != "true":
            core.evaluate(n_episodes=1, render=True, quiet=True)
            os.system(f"ffmpeg -y -r 50 -i {custom_tmp_path}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p {fname} > /dev/null 2>&1")
    except KeyError:
        return


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


def set_mujoco_custom_rendering(enabled: bool):
    """
    Set the environment variable RENDER to "true" or "false" to enable the custom mushroom_rl core code for rendering.
    :param enabled: True to enable, False to disable.
    :return: None
    """
    os.environ["RENDER"] = "true" if enabled else "false"


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