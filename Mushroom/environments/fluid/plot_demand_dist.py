import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from Mushroom.environments.fluid.circular_network import CircularFluidNetwork

num_samples = 1_000_000
bins = 1000
samples = np.zeros((num_samples, 4))

for i in tqdm(range(num_samples)):
    samples[i, :] = CircularFluidNetwork._configure_demand("uniform_individual", 0.3, 1.5, 4)

plt.hist(samples.sum(1), bins=bins, density=False)
plt.ylim((0, num_samples // bins * 2))
plt.show()

for i in range(4):
    plt.hist(samples[:, i], bins=bins, density=False, alpha=0.25, label=f"Valve {i}")
plt.ylim((0, num_samples // bins * 3))
plt.legend()
plt.show()

print(samples.max(0))
print(samples.min(0))
