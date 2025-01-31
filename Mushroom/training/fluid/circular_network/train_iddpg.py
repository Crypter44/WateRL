import json

import numpy as np
import matplotlib as mpl
from mushroom_rl.utils.dataset import compute_metrics
from tqdm import tqdm

from Mushroom.agents.agent_factory import create_ddpg_agent
from Mushroom.agents.sigma_decay_policies import set_noise_for_all, update_sigma_for_all
from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Mushroom.core.multi_agent_core import MultiAgentCore
from Mushroom.utils.plotting import plot_training_data
from Mushroom.utils.utils import set_seed, parametrized_training

# PARAMS
gamma = 0.99
gamma_eval = 1.

lr_actor = 5e-5
lr_critic = 1e-4

initial_replay_size = 1000
max_replay_size = 50000
batch_size = 200

n_features = 80
tau = .005

sigma = [(0, 0.3), (75, 0.2), (150, 0.1)]

n_epochs = 200
n_steps_learn = 1400
n_steps_test = 600
n_steps_per_fit = 1

num_agents = 2
n_episodes_final = 500
n_episodes_final_render = 100
n_epochs_per_checkpoint = 100

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
    "negative_flow": {"w": 1}
}
demand = ("uniform_global", 0.4, 1.4)
# END_PARAMS

mpl.rcParams['figure.max_open_warning'] = -1


# create a dictionary to store data for each seed
def train(p1, p2, seed, save_path):
    set_seed(seed)
    # MDP
    mdp = CircularFluidNetwork(gamma=gamma, criteria=criteria, demand=demand)
    agents = [create_ddpg_agent(
        mdp,
        agent_idx=i,
        n_features_actor=n_features,
        lr_actor=lr_actor,
        n_features_critic=n_features,
        lr_critic=lr_actor,
        batch_size=batch_size,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        tau=tau,
        sigma_checkpoints=sigma,
        decay_type="linear",
    ) for i in range(num_agents)]

    # Core
    core = MultiAgentCore(agent=agents, mdp=mdp)

    core.learn(
        n_steps=initial_replay_size,
        n_steps_per_fit_per_agent=[initial_replay_size] * num_agents,
        quiet=True
    )
    data = [compute_metrics(core.evaluate(n_steps=n_steps_test, render=False, quiet=True), gamma_eval)]
    core.mdp.render(save_path=save_path + f"Epoch_0")

    pbar = tqdm(range(n_epochs), unit='epoch', leave=False)
    for n in pbar:
        core.learn(
            n_steps=n_steps_learn,
            n_steps_per_fit_per_agent=[n_steps_per_fit] * num_agents,
            quiet=True
        )

        core.evaluate(n_steps=200, render=False, quiet=True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}_Noisy")
        set_noise_for_all(core.agent, False)
        dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
        set_noise_for_all(core.agent, True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}")

        data.append(compute_metrics(dataset, gamma_eval))
        pbar.set_postfix(
            MinMaxMean=np.round(data[-1][0:3], 2),
            sigma=np.round(core.agent[0].policy._sigma, 2)
        )

        update_sigma_for_all(agents)

        if (n + 1) % n_epochs_per_checkpoint == 0:
            for i, a in enumerate(core.agent):
                a.save(save_path + f"/checkpoints/Epoch_{n + 1}_Agent_{i}")

    set_noise_for_all(core.agent, False)
    for i in range(n_episodes_final_render):
        core.evaluate(n_episodes=1, quiet=True)
        core.mdp.render(save_path=save_path + f"Final_{i}")

    for i, a in enumerate(core.agent):
        a.save(save_path + f"/checkpoints/Final_Agent_{i}")

    if n_episodes_final > 0:
        with open(save_path + "Evaluation.json", "w") as f:
            final = compute_metrics(
                core.evaluate(n_episodes=n_episodes_final, render=False, quiet=False),
                gamma_eval
            )
            json.dump({
                "Min": final[0],
                "Max": final[1],
                "Mean": final[2],
                "Median": final[3],
                "Count": final[4],
                "Took:": pbar.format_dict["elapsed"]
            }, f, indent=4)

    return {"metrics": np.array(data), "additional_data": {}}


training_data, path = parametrized_training(
    __file__,
    [None],
    [None],
    [1],
    train=train,
    base_path="Plots/IDDPG/",
)

plot_training_data(training_data, path)
