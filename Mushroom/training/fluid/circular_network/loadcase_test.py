import json
import os

import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.core import Agent
from tqdm import tqdm

from Mushroom.agents.sigma_decay_policies import set_noise_for_all
from Mushroom.core.multi_agent_core_labeled import MultiAgentCoreLabeled
from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Mushroom.utils.utils import compute_metrics_with_labeled_dataset

num_agents = 2

path = input("Enter the path to the first agent: ")
save_path = "./LoadcasePlots/" + input("Enter the name of the folder to save the test results: ")
os.makedirs(save_path, exist_ok=True)

state_selector = [
    0, 1, 2, 3
]

obs_selector = [
    [0, 1],
    [2, 3],
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

criteria = {
    "demand": {
        "w": 10.0,
        "bound": 0.1,
        "value_at_bound": 0.001,
        "max": 0,
        "min": -1
    },
    "power_per_flow": {"w": .06},
    "negative_flow": {
        "w": 2.0,
        "threshold": -1e-6,
    },
    "target_opening": {
        "max": 1,
        "min": 0,
        "w": 1.0,
        "left_bound": 0.5,
        "value_at_left_bound": 0.001,
        "right_bound": 0.04,
        "value_at_right_bound": 0.001,
    }
}

mdp = CircularFluidNetwork(
    labeled_step=True,
    multi_threaded_rendering=False,
    plot_rewards=False,
    observation_selector=obs_selector,
    state_selector=state_selector,
    criteria=criteria,
)
agents = []
for i in range(num_agents):
    p = path[:-1] + str(i)
    print(f"Attempting to load agent from: \n{p}")
    agents.append(Agent.load(p))

core = MultiAgentCoreLabeled(agents, mdp)
set_noise_for_all(agents, False)

powers = []
openings = []
deviations = []
for tc in tqdm(test_cases):
    mdp.reset(demand=("test", 0.4, 1.4, tc))
    dataset, _ = core.evaluate(n_episodes=1, quiet=True)
    results = mdp.sim.get_results()
    combined_maximum_power_consumption = max(results["water_network.P_pum_1"]) + max(results["water_network.P_pum_4"])

    maximum_valve_opening = max(
        np.array(results["water_network.u_v_2"])[-1],
        np.array(results["water_network.u_v_3"])[-1],
        np.array(results["water_network.u_v_5"])[-1],
        np.array(results["water_network.u_v_6"])[-1],
    )

    deviation_from_demand = max(
        np.abs(np.array(results["water_network.V_flow_2"])[-1] - tc[0]),
        np.abs(np.array(results["water_network.V_flow_3"])[-1] - tc[1]),
        np.abs(np.array(results["water_network.V_flow_5"])[-1] - tc[2]),
        np.abs(np.array(results["water_network.V_flow_6"])[-1] - tc[3]),
    )

    powers.append(combined_maximum_power_consumption)
    openings.append(maximum_valve_opening)
    deviations.append(deviation_from_demand)

    score = compute_metrics_with_labeled_dataset(dataset)
    mdp.render(
        title=f"Test with total: {np.round(sum(tc), 2)},"
              f" combined power: {np.round(combined_maximum_power_consumption, 2)},"
              f" max valve opening: {np.round(maximum_valve_opening, 2)},"
              f" deviation from demand: {np.round(deviation_from_demand, 2)}"
              f" score: {np.round(score[0], 2)}",
        save_path=f"{save_path}/{len(powers) - 1}",
    )

np.save(f"{save_path}/powers", powers)
np.save(f"{save_path}/openings", openings)
np.save(f"{save_path}/deviations", deviations)
np.savetxt(f"{save_path}/powers", powers)
np.savetxt(f"{save_path}/openings", openings)
np.savetxt(f"{save_path}/deviations", deviations)
