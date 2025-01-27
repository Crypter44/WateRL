from mushroom_rl.core import Agent

from Mushroom.core.multi_agent_core import MultiAgentCore
from Mushroom.environments.fluid.circular_network import CircularFluidNetwork

# PARAMS
num_agents = 2
gamma = 0.99
gamma_eval = 1.

path = """/Mushroom/training/fluid/circular_network/Plots/IDDPG/25-01-22__08:55/None-None/s0/checkpoints/Final_Agent_0"""

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

# END_PARAMS

mdp = CircularFluidNetwork(gamma=gamma, criteria=criteria)
agents = []
for i in range(num_agents):
    p = path[:-2] + str(i)
    print(p)
    agents.append(Agent.load(p))

core = MultiAgentCore(agents, mdp)

core.evaluate(n_episodes=1)
core.mdp.render()
