import json

import matplotlib as mpl
import numpy as np
from tqdm import tqdm

from Mushroom.agents.agent_factory import create_ddpg_agent
from Mushroom.agents.sigma_decay_policies import set_noise_for_all, update_sigma_for_all
from Mushroom.core.multi_agent_core_labeled import MultiAgentCoreLabeled
from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Mushroom.utils.plotting import plot_training_data, plot_debug_data
from Mushroom.utils.utils import set_seed, parametrized_training, compute_metrics_with_labeled_dataset, final_evaluation

# PARAMS
gamma = 0.99
gamma_eval = 1.

lr_actor = 5e-5
lr_critic = 1e-4

initial_replay_size_episodes = 5
max_replay_size_episodes = 5000
batch_size = 200

n_features = 80
tau = .005

sigma = [(0, 0.7), (20, 0.3), (300, 0.1)]

n_epochs = 30
n_episodes_learn = 10
n_episodes_test = 3
n_steps_per_fit = 1

num_agents = 2
n_episodes_final = 1000
n_episodes_final_render = 200
n_epochs_per_checkpoint = 100

criteria = {
    "demand": {
        "w": 10.0,
        "bound": 0.1,
        "value_at_bound": 0.001,
    },
    "power_per_flow": {"w": 0.1},
    "negative_flow": {"w": 1.0},
}
demand = ("uniform_global", 0.4, 1.4)
# END_PARAMS

mpl.rcParams['figure.max_open_warning'] = -1


# create a dictionary to store data for each seed
def train(p1, p2, seed, save_path):
    set_seed(seed)
    # MDP
    mdp = CircularFluidNetwork(gamma=gamma, criteria=criteria, demand=demand, labeled_step=True)
    agents = [create_ddpg_agent(
        mdp,
        agent_idx=i,
        n_features_actor=n_features,
        lr_actor=lr_actor,
        n_features_critic=n_features,
        lr_critic=lr_actor,
        batch_size=batch_size,
        initial_replay_size=initial_replay_size_episodes * mdp.info.horizon,
        max_replay_size=max_replay_size_episodes * mdp.info.horizon,
        tau=tau,
        sigma_checkpoints=sigma,
        decay_type="linear",
    ) for i in range(num_agents)]
    debug_infos = [None for _ in range(num_agents)]
    for a in agents:
        a.set_debug_logging(True)

    # Core
    core = MultiAgentCoreLabeled(agents=agents, mdp=mdp)

    core.learn(
        n_steps=initial_replay_size_episodes * mdp.info.horizon,
        n_steps_per_fit_per_agent=[initial_replay_size_episodes * mdp.info.horizon] * num_agents,
    )
    set_noise_for_all(core.agents, False)
    data = [compute_metrics_with_labeled_dataset(core.evaluate(n_episodes=n_episodes_test, render=False)[0])]
    set_noise_for_all(core.agents, True)
    core.mdp.render(save_path=save_path + f"Epoch_0")

    pbar = tqdm(range(n_epochs), unit='epoch', leave=False)
    for n in pbar:
        core.learn(
            n_episodes=n_episodes_learn,
            n_steps_per_fit_per_agent=[n_steps_per_fit] * num_agents,
        )
        for i, a in enumerate(agents):
            debug_infos[i] = a.get_debug_info(debug_infos[i])

        core.evaluate(n_episodes=1, render=False, quiet=True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}_Noisy")
        set_noise_for_all(core.agents, False)
        dataset, _ = core.evaluate(n_episodes=n_episodes_test, render=False, )
        set_noise_for_all(core.agents, True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}")

        data.append(compute_metrics_with_labeled_dataset(dataset))
        pbar.set_postfix(
            MinMaxMean=np.round(data[-1][0:3], 2),
            sigma=np.round(core.agents[0].policy.get_sigma(), 2)
        )

        update_sigma_for_all(agents)

        if (n + 1) % n_epochs_per_checkpoint == 0:
            for i, a in enumerate(core.agents):
                a.save(save_path + f"/checkpoints/Epoch_{n + 1}_Agent_{i}")

    set_noise_for_all(core.agents, False)
    plot_debug_data(debug_infos[0], save_path)
    final_evaluation(n_episodes_final, n_episodes_final_render, core, save_path)

    return {"metrics": np.array(data), "additional_data": {}}


training_data, path = parametrized_training(
    __file__,
    [None],
    [None],
    [0],
    train=train,
    base_path="./Plots/IDDPG/",
)

plot_training_data(training_data, path)
