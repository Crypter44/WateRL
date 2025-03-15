import numpy as np
from matplotlib import pyplot as plt

done = False
paths = []
while not done:
    path = input("Enter path to the first test result: ")
    paths.append(path)
    user_input = input("Done? (y/n) ")
    while user_input not in ["y", "n"]:
        user_input = input("Done? (y/n) ")
    if user_input == "y":
        done = True

num_tests = -1
test_results = []
for p in paths:
    test_result = np.load(p)
    if num_tests == -1:
        num_tests = len(test_result)
    elif num_tests != len(test_result):
        raise ValueError("All test results must have the same number of test cases.")
    test_results.append(test_result)

# Number of runs
num_runs = len(paths)
# Width of each bar
bar_width = 0.6 / num_runs
# Positions of the bars
positions = np.arange(num_tests)

# Determine the best run for each test
best_runs = np.argmin(test_results, axis=0)

# Count the number of times each run had the best score and get the test IDs
best_counts = np.bincount(best_runs, minlength=num_runs)
best_test_ids = [np.where(best_runs == i)[0] for i in range(num_runs)]

plt.figure(figsize=(8 * num_runs, 8), dpi=150)
for i, test_result in enumerate(test_results):
    label = f"{paths[i].split('/')[-2]} ({best_counts[i]} best scores: {list(best_test_ids[i])})"
    bars = plt.bar(positions + i * bar_width, test_result, bar_width, alpha=0.75, label=label)
    for j, bar in enumerate(bars):
        if best_runs[j] == i:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, '*', ha='center', va='center',
                     fontsize=12, color='black')

plt.xlabel("Test case")
plt.xticks(positions + bar_width * (num_runs - 1) / 2, range(num_tests))
plt.ylabel("Combined power consumption [W]")
plt.ylim((0, 300))
plt.title("Combined power consumption for each test case")
plt.legend(loc="upper left")

plt.show()
