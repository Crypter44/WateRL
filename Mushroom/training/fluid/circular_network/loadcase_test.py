import json
import os

import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.core import Agent
from tqdm import tqdm

from Mushroom.core.multi_agent_core_labeled import MultiAgentCoreLabeled
from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Mushroom.utils.utils import compute_metrics_with_labeled_dataset

num_agents = 2

path = input("Enter the path to the first agent: ")
save_path = "./LoadcasePlots/" + input("Enter the name of the folder to save the test results: ")
os.makedirs(save_path, exist_ok=True)

obs_selectors = [
    [0, 1, 2],
    [1, 2, 3],
]

test_cases = [
    # Minimum load cases (1.6):
    (0.4, 0.4, 0.4, 0.4),

    # Low load cases (2.6):
    (0.65, 0.65, 0.65, 0.65),
    (1.4, 0.4, 0.4, 0.4),
    (0.4, 1.4, 0.4, 0.4),
    (0.4, 0.4, 1.4, 0.4),
    (0.4, 0.4, 0.4, 1.4),

    # Medium load cases (3.6):
    (0.9, 0.9, 0.9, 0.9),
    (1.4, 1.4, 0.4, 0.4),
    (0.4, 1.4, 1.4, 0.4),
    (0.4, 0.4, 1.4, 1.4),
    (1.4, 0.4, 0.4, 1.4),
    (0.4, 1.4, 0.4, 1.4),
    (1.4, 0.4, 1.4, 0.4),

    # High load cases (4.6):
    (1.15, 1.15, 1.15, 1.15),
    (1.4, 1.4, 1.4, 0.4),
    (1.4, 1.4, 0.4, 1.4),
    (1.4, 0.4, 1.4, 1.4),
    (0.4, 1.4, 1.4, 1.4),

    # Maximum load cases (5.6):
    (1.4, 1.4, 1.4, 1.4),
]


mdp = CircularFluidNetwork(
    labeled_step=True,
    multi_threaded_rendering=False,
    plot_rewards=False,
    observation_selectors=obs_selectors
)
agents = []
for i in range(num_agents):
    p = path[:-1] + str(i)
    print(f"Loaded agent from: {p}")
    agents.append(Agent.load(p))

core = MultiAgentCoreLabeled(agents, mdp)

powers = []
for tc in tqdm(test_cases):
    mdp.reset(demand=("test", 0.4, 1.4, tc))
    dataset, _ = core.evaluate(n_episodes=1, quiet=True)
    results = mdp.sim.get_results()
    combined_power = max(results["water_network.P_pum_1"]) + max(results["water_network.P_pum_4"])
    powers.append(combined_power)
    score = compute_metrics_with_labeled_dataset(dataset)
    mdp.render(
        title=f"Test with total: {np.round(sum(tc), 2)}, combined power: {np.round(combined_power, 2)}",
        save_path=f"{save_path}/{len(powers) - 1}",
    )

np.save(f"{save_path}/results", powers)
