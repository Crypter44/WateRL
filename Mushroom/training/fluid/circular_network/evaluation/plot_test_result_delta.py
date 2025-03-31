import os

import numpy as np
from matplotlib import pyplot as plt

num_runs = 2  # Only 2 methods can be visualized at once

# Configure which criterion to compare
criterions = input("Enter the criterion compared (power, opening, deviation, ALL): ")
if criterions == "":
    criterions = "all"
if criterions.lower() not in ["power", "opening", "deviation", "all"]:
    raise ValueError("Invalid criterion. Choose between 'power', 'opening' and 'deviation' or 'all'.")

# Create a dictionary to store the test results for each criterion
test_results = {
    'power': [],
    'opening': [],
    'deviation': []
}
labels = []

criterions = ["power", "opening", "deviation"] if criterions == "all" else [criterions]
for num_run in range(int(num_runs)):
    path = input(f"Enter the path to the test results for method {num_run + 1}: ")
    labels.append(input(f"Enter the label for method {num_run + 1}: "))

    # check if the path has subdirectories
    average = False
    for file in os.listdir(path):
        if not average and os.path.isdir(os.path.join(path, file)):
            average = True

            input("The path contains subdirectories. All subdirectories will be averaged. \nPress Enter to continue.")
        elif average and not os.path.isdir(os.path.join(path, file)):
            raise ValueError("The path contains both directories and files. Incorrect input format.")

    for criterion in criterions:
        if average:
            test_result = None
            for i, folder in enumerate(os.listdir(path)):
                try:
                    if i == 0:
                        loaded = np.load(os.path.join(path, folder) + f"/{criterion}s.npy")
                        num_tests = loaded.shape[0]
                        test_result = np.zeros((num_tests, len(os.listdir(path))))
                    test_result[:, i] = np.load(os.path.join(path, folder) + f"/{criterion}s.npy")
                except FileNotFoundError:
                    raise FileNotFoundError(f"'{criterion}s.npy' not found."
                                            f"Make sure each subdirectory contains a file named '{criterion}s.npy'.")
            test_result = np.mean(test_result, axis=1)
        else:
            try:
                test_result = np.load(path + f"/{criterion}s.npy")
            except FileNotFoundError:
                raise FileNotFoundError(f"'{criterion}s.npy' not found. "
                                        f"Make sure the directory contains a file named '{criterion}s.npy'.")
        test_results[criterion].append(test_result)

# determine smoothing kernel size
kernel_input = input("Enter the kernel size for smoothing the difference (0 for no smoothing): ")
kernels = [int(kernel_input)] if kernel_input != "0" else []

for criterion in criterions:
    # Plot the difference between the two methods
    plt.figure(figsize=(16, 8))
    plt.axhline(0, color='black', linewidth=1, linestyle='--')

    x = np.arange(len(test_results[criterion][0]))
    diff = test_results[criterion][0] - test_results[criterion][1]
    smoothed_diff = diff
    for k in kernels:
        # create a discrete gaussian kernel
        gaussian_kernel = np.exp(-np.linspace(-k, k, 2 * k + 1) ** 2 / (2 * k ** 2))
        gaussian_kernel /= np.sum(gaussian_kernel)
        padded_diff = np.pad(smoothed_diff, (k, k), mode='edge')
        smoothed_diff = np.convolve(padded_diff, gaussian_kernel, mode='valid')
    # color the plot according to the sign of the difference
    plt.plot(x, diff, label='Difference', alpha=0.3, color='black')

    plt.fill_between(
        x, 0, diff,
        where=diff < 0 if criterion in ["power", "deviation"] else diff > 0,
        color='red',
        alpha=0.1,
        interpolate=True,
        label=labels[0] + " is better"
    )

    plt.fill_between(
        x, 0, diff,
        where=diff > 0 if criterion in ["power", "deviation"] else diff < 0,
        color='blue',
        alpha=0.1,
        interpolate=True,
        label=labels[1] + " is better"
    )

    plt.plot(smoothed_diff, label='Smoothed difference', color='black')

    if criterion == "power":
        plt.ylabel("Difference in power consumption [W]")
    elif criterion == "opening":
        plt.ylabel("Difference in valve opening")
    elif criterion == "deviation":
        plt.ylabel("Difference in deviation [m³/h]")

    plt.title(f"Difference in {criterion} between {labels[0]} and {labels[1]}")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=1)
    plt.subplots_adjust(left=0.075, bottom=0.3, right=0.925, top=0.9)
    plt.show()
