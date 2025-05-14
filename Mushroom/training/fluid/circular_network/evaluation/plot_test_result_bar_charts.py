import os

import numpy as np
from matplotlib import pyplot as plt

num_runs = int(input("Enter the number of methods to compare: "))
num_test_input = input("Enter the number of test cases (19): ")
num_tests = 19 if num_test_input == "" else int(num_test_input)
criterias = input("Enter the criteria compared (power, opening, deviation, ALL): ")
if criterias == "":
    criterias = "all"
if criterias.lower() not in ["power", "opening", "deviation", "all"]:
    raise ValueError("Invalid criteria. Choose between 'power', 'opening' and 'deviation' or 'all'.")
test_results = {
    'power': [],
    'opening': [],
    'deviation': []
}
labels = []

criterias = ["power", "opening", "deviation"] if criterias == "all" else [criterias]
for num_run in range(int(num_runs)):
    path = input(f"Enter the path to the test results for method {num_run + 1}: ")
    label_input = input(f"Enter the label for method {num_run + 1}: ")
    if label_input == "":
        label_input = path.split("/")[-1]
    labels.append(label_input)

    # check if the path has subdirectories
    average = False
    for file in os.listdir(path):
        if not average and os.path.isdir(os.path.join(path, file)):
            average = True
            input("The path contains subdirectories. All subdirectories will be averaged. \nPress Enter to continue.")
        elif average and not os.path.isdir(os.path.join(path, file)):
            raise ValueError("The path contains both directories and files. Incorrect input format.")

    for criteria in criterias:
        if average:
            test_result = np.zeros((num_tests, len(os.listdir(path))))
            for i, folder in enumerate(os.listdir(path)):
                excludes = [".DS_Store", ]
                if folder in excludes:
                    continue
                try:
                    test_result[:, i] = np.load(os.path.join(path, folder) + f"/{criteria}s.npy")
                except FileNotFoundError:
                    print(f"'{criteria}s.npy' not found."
                          f"Make sure each subdirectory contains a file named '{criteria}s.npy'.")
                except Exception as e:
                    print(f"Error loading '{criteria}s.npy' from {folder}.")
                    print(e)
            test_result = np.mean(test_result, axis=1)
        else:
            try:
                test_result = np.load(path + f"/{criteria}s.npy")
            except FileNotFoundError:
                raise FileNotFoundError(f"'{criteria}s.npy' not found. "
                                        f"Make sure the directory contains a file named '{criteria}s.npy'.")
        test_results[criteria].append(test_result)

for criteria in criterias:
    # Number of runs
    # Width of each bar
    bar_width = 0.8 / num_runs
    # Positions of the bars
    positions = np.arange(num_tests)

    # Determine the best run for each test
    if criteria == "power" or criteria == "deviation":
        best_runs = np.argmin(test_results[criteria], axis=0)
    else:
        best_runs = np.argmax(test_results[criteria], axis=0)

    # Count the number of times each run had the best score and get the test IDs
    best_counts = np.bincount(best_runs, minlength=num_runs)
    best_test_ids = [np.where(best_runs == i)[0] for i in range(num_runs)]

    plt.figure(figsize=(10 + 0.2 * num_runs, 6), dpi=150)
    for i, test_result in enumerate(test_results[criteria]):
        label = f"{labels[i]} ({best_counts[i]} best scores: {list(best_test_ids[i])})"
        bars = plt.bar(positions + i * bar_width, test_result, bar_width, alpha=0.75, label=label)
        for j, bar in enumerate(bars):
            if j in [0, 5, 12, 17] and i == 3:
                plt.axvline(bar.get_x() + bar.get_width() * 1.5, color='black', linewidth=1, linestyle='--')
            if best_runs[j] == i:
                plt.scatter(bar.get_x() + bar.get_width() / 2, bar.get_height(), marker='v', color='black', s=20,
                            label="Best method per test" if j == 0 else "")

    plt.xlabel("Test case")
    plt.xticks(positions + bar_width * (num_runs - 1) / 2, range(num_tests))

    if criteria == "power":
        plt.ylabel("Combined power consumption [W]")
        plt.ylim((0, 300))
    elif criteria == "opening":
        plt.ylabel("Opening")
        plt.ylim((0, 1))
    elif criteria == "deviation":
        # make y axis logarithmic
        plt.yscale('log')
        plt.ylabel("Deviation (lower = better) [m³/h]")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=1)
    plt.subplots_adjust(left=0.1, bottom=0.4, right=0.975, top=0.975)
    plt.savefig(f"comparison_{criteria}.pdf")
    plt.show()
