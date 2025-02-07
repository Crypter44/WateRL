import json

from mushroom_rl.core import Agent

from Mushroom.core.multi_agent_core import MultiAgentCore
from Mushroom.core.multi_agent_core_labeled import MultiAgentCoreLabeled
from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Mushroom.utils.utils import compute_metrics_with_labeled_dataset

# PARAMS
num_agents = 2
gamma = 0.99
gamma_eval = 1.

path = \
"/home/js/PycharmProjects/Thesis/Mushroom/training/fluid/circular_network/Plots/MADDPG/PowerPerFlow[Short]/None-None/s1/checkpoints/Final_Agent_0"

criteria = {
    "target_opening": {
        "w": 1.,
        "target": 0.95,
        "smoothness": 0.0001,
        "left_bound": 0.2,
        "value_at_left_bound": 0.05,
        "right_bound": 0.05,
        "value_at_right_bound": 0.001,
    },
}

n_renders = 10
n_episodes_test = 500

# END_PARAMS

mdp = CircularFluidNetwork(gamma=gamma, criteria=criteria, labeled_step=True)
agents = []
for i in range(num_agents):
    p = path[:-1] + str(i)
    print(p)
    agents.append(Agent.load(p))

core = MultiAgentCoreLabeled(agents, mdp)

for i in range(n_renders):
    core.evaluate(n_episodes=1, quiet=True)
    core.mdp.render(save_path=None)

if n_episodes_test > 0:
    with open("Evaluation.json", "w") as f:
        final = compute_metrics_with_labeled_dataset(
            core.evaluate(n_episodes=n_episodes_test, render=False, quiet=False)[0],
            gamma_eval
        )
        json.dump({
            "Min": final[0],
            "Max": final[1],
            "Mean": final[2],
            "Median": final[3],
            "Count": final[4],
        }, f, indent=4)