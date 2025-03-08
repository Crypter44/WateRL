import json

import matplotlib as mpl
import numpy as np
from tqdm import tqdm

from Mushroom.agents.agent_factory import setup_maddpg_agents
from Mushroom.agents.sigma_decay_policies import set_noise_for_all, update_sigma_for_all
from Mushroom.core.multi_agent_core_labeled import MultiAgentCoreLabeled
from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Mushroom.utils.plotting import plot_training_data, plot_debug_data
from Mushroom.utils.utils import set_seed, parametrized_training, compute_metrics_with_labeled_dataset, final_evaluation

# PARAMS
gamma = 0.99
gamma_eval = 1.

lr_actor = 2e-3
lr_critic = 4e-3

initial_replay_size = 1000
max_replay_size = 50000
batch_size = 200

n_features = 80
tau = .005

sigma = [(0, 0.7), (20, 0.5), (40, 0.1)]
decay_type = "exponential"

n_epochs = 50
n_steps_learn = 1400
n_steps_test = 600
n_steps_per_fit = 1

num_agents = 2
n_episodes_final = 500
n_episodes_final_render = 100
n_epochs_per_checkpoint = 100

criteria = {
    "demand": {
        "w": 10.0,
        "bound": 0.1,
        "value_at_bound": 0.001,
    },
    "power_per_flow": {"w": 0.175},
    "negative_flow": {"w": 1.0},
}
demand = ("uniform_global", 0.4, 1.4)
# END_PARAMS

mpl.rcParams['figure.max_open_warning'] = -1


# create a dictionary to store data for each seed
def train(p1, p2, seed, save_path):
    length = n_epochs

    set_seed(seed)
    # MDP
    mdp = CircularFluidNetwork(gamma=gamma, criteria=criteria, demand=demand, labeled_step=True)
    agents = setup_maddpg_agents(
        n_agents=num_agents,
        mdp=mdp,
        n_features_actor=n_features,
        lr_actor=lr_actor,
        n_features_critic=n_features,
        lr_critic=lr_actor,
        batch_size=batch_size,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        tau=tau,
        sigma_checkpoints=sigma,
        decay_type=decay_type,
    )
    debug_infos = [None for _ in range(num_agents)]
    for a in agents:
        a.set_debug_logging(True)

    # Core
    core = MultiAgentCoreLabeled(agents=agents, mdp=mdp)

    set_noise_for_all(core.agents, False)
    data = [compute_metrics_with_labeled_dataset(core.evaluate(n_steps=n_steps_test, render=False, quiet=True)[0])]
    core.mdp.render(save_path=save_path + f"Epoch_0")
    set_noise_for_all(core.agents, True)
    core.evaluate(n_episodes=1, render=False, quiet=True)
    core.mdp.render(save_path=save_path + f"Epoch_0_Noisy")

    core.learn(
        n_steps=initial_replay_size,
        n_steps_per_fit_per_agent=[initial_replay_size] * num_agents,
    )

    pbar = tqdm(range(length), unit='epoch', leave=False)
    for n in pbar:
        core.learn(
            n_steps=n_steps_learn,
            n_steps_per_fit_per_agent=[n_steps_per_fit] * num_agents,
        )
        for i, a in enumerate(agents):
            debug_infos[i] = a.get_debug_info(debug_infos[i])

        core.evaluate(n_steps=200, render=False, quiet=True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}_Noisy")
        set_noise_for_all(core.agents, False)
        dataset, _ = core.evaluate(n_steps=n_steps_test, render=False)
        set_noise_for_all(core.agents, True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}")

        data.append(compute_metrics_with_labeled_dataset(dataset, gamma_eval))
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
    [1],
    train=train,
    base_path="Plots/MADDPG/",
)

plot_training_data(training_data, path)
