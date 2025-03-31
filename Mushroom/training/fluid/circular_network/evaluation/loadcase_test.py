import json
import os

import numpy as np
from mushroom_rl.core import Agent
from tqdm import tqdm

from Mushroom.agents.sigma_decay_policies import set_noise_for_all
from Mushroom.core.multi_agent_core_labeled import MultiAgentCoreLabeled
from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Mushroom.training.fluid.circular_network.evaluation.loadcase_generator import gen_tests_for_visual_inspection
from Mushroom.utils.utils import compute_metrics_with_labeled_dataset

#  ---------------- User input for configuring the tests ----------------
# 1. Load the test config
config_input = input("Enter the path to the test config: ")
config = json.load(open(config_input))

# 2. Load the agent(s) and the path to save the test results
path_input = input("Enter the path to the first agent: ")
save_path_input = "./LoadcaseTestResults/" + input("Enter the name of the folder to save the test results: ")
paths = []
save_paths = []
if os.path.isdir(path_input):
    print("The given agent path is a directory.")

    if "checkpoints" in path_input.split("/")[-1]:
        print("Detected a checkpoint folder. Running tests for all agents in this training.")
        input("Press Enter to continue...")
        for p in os.listdir(path_input):
            if p[-1] == "0":
                epoch_count = p.split("_")[1]
                paths.append(os.path.join(path_input, p))
                save_paths.append(os.path.join(save_path_input, epoch_count))
        print(f"Found {len(paths)} agents.")
    else:
        print("The directory does not contain checkpoints."
              " It will be treated as the root directory of a parametrized training.")
        print("Running tests for the agents in the subdirectories.")
        input("Press Enter to continue...")
        agent = input("What stage of the training do you want to test? (<Epoch number> or 'Final'): ")
        if agent == "Final":
            agent = "Final_Agent_0"
        else:
            agent = f"Epoch_{agent}_Agent_0"
        for p in os.listdir(path_input):
            if os.path.isdir(os.path.join(path_input, p)):
                if p.count("wandb") == 0:
                    save_paths.append(os.path.join(save_path_input, p.split("/")[-1]))
                    paths.append(os.path.join(path_input, p) + "/checkpoints/" + agent)

else:
    paths.append(path_input)
    save_paths.append(save_path_input)

for save_path in save_paths:
    os.makedirs(save_path, exist_ok=True)

# 3. Load the test cases
test_case_input = input("Enter the path to the test cases or leave empty (Visual Test Set): ")
if test_case_input == "":
    test_cases = gen_tests_for_visual_inspection()
else:
    test_cases = np.load(test_case_input)

#  ---------------- Run the tests ----------------
accepted_agents = []
for path, save_path in zip(paths, save_paths):
    mdp = CircularFluidNetwork(
        labeled_step=True,
        multi_threaded_rendering=False,
        plot_rewards=False,
        observation_selector=config["obs_selector"],
        state_selector=config["state_selector"],
        criteria=config["criteria"],
    )
    agents = []
    for i in range(2):
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

# ---------------- Save the test results ----------------
    print("--------------------- Test Results ---------------------")
    print(f"Deviation - max: {np.round(max(deviations), 4)}, min: {np.round(min(deviations), 4)}, "
          f"avg: {np.round(np.mean(deviations), 4)}")
    print(f"Power - max: {np.round(max(powers), 2)}, min: {np.round(min(powers), 2)}, "
          f"avg: {np.round(np.mean(powers), 2)}")
    print(f"Opening - max: {np.round(max(openings), 3)}, min: {np.round(min(openings), 3)}, "
          f"avg: {np.round(np.mean(openings), 3)}")
    print("--------------------------------------------------------")
    if max(openings) < 1.0:
        print(f"Test passed!")
        accepted_agents += [save_path]
    else:
        print(f"Test failed! There was deviation from the demand, that was caused by the pumps!")
    print("--------------------------------------------------------")
    print()

    np.save(f"{save_path}/powers", powers)
    np.save(f"{save_path}/openings", openings)
    np.save(f"{save_path}/deviations", deviations)
    np.save(f"{save_path}/test_cases", test_cases)
    np.savetxt(f"{save_path}/powers", powers)
    np.savetxt(f"{save_path}/openings", openings)
    np.savetxt(f"{save_path}/deviations", deviations)
    np.savetxt(f"{save_path}/test_cases", test_cases)

print(f"Accepted {len(accepted_agents)} out of {len(paths)} agents.")
with open(f"{save_path_input}/accepted_agents.txt", "w") as f:
    for agent in sorted(accepted_agents):
        f.write(agent + "\n")
