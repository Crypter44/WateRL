import numpy as np
from matplotlib import pyplot as plt


def _plot_metrics_to_ax(ax, data: dict, title: str, range_alpha=0.1, color=None):
    for seed, data_for_seed in data.items():
        metrics = data_for_seed["metrics"]
        ax.plot(metrics[:, 2], label=f"Mean score, s={seed}", color=color)
        ax.fill_between(np.arange(len(metrics)), metrics[:, 0], metrics[:, 1], alpha=range_alpha,
                        label=f"Range, s={seed}", color=color)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend(loc='lower right')


def plot_additional_data_to_ax(ax, data: dict, title: str, list_of_data_names_in_one_plot: list):
    for seed, data_for_seed in data.items():
        for data_name in list_of_data_names_in_one_plot:
            data = data_for_seed["additional_data"][data_name]
            ax.plot(data, label=f"{data_name}, s={seed}")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()


def plot_training_data(
        plotting_data: dict,
        path: str,
        plot_additional_data: bool = False,
):
    """
    Plot the training data.

    If multiple_seeds_per_plot is True,
    then all the data from a single experiment will be plotted on the same plot.
    Otherwise, each seed will be plotted on a separate plot.

    The expected structure of the plotting_data dictionary is as follows:
    {
        p1: {
            p2: {
                seed: {
                    "metrics": [(min, max, mean, median, count), ...],
                    "additional_data": {
                        "data_name": [data, ...],
                    }
                }
            }
        }
    }

    :param plotting_data: A dictionary containing the training data.
    :param path: The path to save the plot.
    :param plot_additional_data: Whether to plot additional data.
    :return: None
    """

    p1s = list(plotting_data.keys())
    p2s = list(list(plotting_data.values())[0].keys())
    seeds = list(list(list(plotting_data.values())[0].values())[0].keys())

    rows = len(p1s)
    cols = len(p2s)

    left_padding = 0.2 / cols
    bottom_padding = 0.2 / rows
    right_padding = 1 - left_padding
    top_padding = 1 - bottom_padding
    hspace_padding = 0.75 / rows
    wspace_padding = 0.75 / cols

    fig, ax = plt.subplots(rows, cols, figsize=((cols + 2 * left_padding) * 5, (rows + 2 * bottom_padding) * 5))

    if rows == 1 and cols == 1:
        _plot_metrics_to_ax(ax, plotting_data[p1s[0]][p2s[0]], "")
    else:
        ax = ax.flatten()
        for i, p1 in enumerate(p1s):
            for j, p2 in enumerate(p2s):
                _plot_metrics_to_ax(ax[i * cols + j], plotting_data[p1][p2], f"p1={p1}, p2={p2}")

    fig.subplots_adjust(
        left=left_padding,
        bottom=bottom_padding,
        right=right_padding,
        top=top_padding,
        hspace=hspace_padding,
        wspace=wspace_padding
    )
    fig.suptitle("Training Metrics")

    plt.savefig(path + "Metrics.png")
    fig.show()
    plt.close(fig)

    if plot_additional_data:
        additional_data_names = list(plotting_data[p1s[0]][p2s[0]][seeds[0]]["additional_data"].keys())

        for data_name in additional_data_names:
            fig, ax = plt.subplots(rows, cols, figsize=((cols + 2 * left_padding) * 5, (rows + 2 * bottom_padding) * 5))
            if rows == 1 and cols == 1:
                plot_additional_data_to_ax(ax, plotting_data[p1s[0]][p2s[0]], "", [data_name])
            else:
                ax = ax.flatten()
                for i, p1 in enumerate(p1s):
                    for j, p2 in enumerate(p2s):
                        plot_additional_data_to_ax(ax[i * cols + j], plotting_data[p1][p2], f"p1={p1}, p2={p2}",
                                                   [data_name])

            fig.subplots_adjust(
                left=left_padding,
                bottom=bottom_padding,
                right=right_padding,
                top=top_padding,
                hspace=hspace_padding,
                wspace=wspace_padding
            )
            fig.suptitle(f"Additional Data: {data_name}")

            plt.savefig(path + f"{data_name}.png")
            fig.show()
            plt.close(fig)
