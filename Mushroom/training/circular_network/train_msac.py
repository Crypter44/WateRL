import numpy as np
from mushroom_rl.utils.dataset import compute_metrics
from tqdm import tqdm

from Mushroom.agents.sac import create_sac_agent
from Mushroom.fluid_network_environments.circular_network import CircularFluidNetwork
from Mushroom.multi_agent_core import MultiAgentCore
from Mushroom.utils import set_seed, plot_data, grid_search

seed = 0

n_features_actor = 80
lr_actor = 1e-3
n_features_critic = 80
lr_critic = 6e-4
batch_size = 200
initial_replay_size = 500
max_replay_size = 10000
warmup_transitions = 0
tau = 0.005
lr_alpha = 0.001
log_std_min = -20
log_std_max = 2
target_entropy = -5

n_epochs = 30
n_steps_learn = 400
n_steps_eval = 400


def train(p1, p2, seed, save_path):
    set_seed(seed)

    mdp = CircularFluidNetwork(gamma=0.99)
    agents = [
        create_sac_agent(
            mdp,
            n_features_actor=n_features_actor,
            lr_actor=lr_actor,
            n_features_critic=n_features_critic,
            lr_critic=lr_critic,
            batch_size=batch_size,
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            warmup_transitions=warmup_transitions,
            tau=tau,
            lr_alpha=lr_alpha,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            target_entropy=target_entropy,
        ) for _ in range(2)
    ]

    core = MultiAgentCore(agent=agents, mdp=mdp)

    data = [compute_metrics(core.evaluate(n_steps=n_steps_eval, render=False, quiet=True))]

    core.evaluate(n_steps=n_steps_eval, render=False, quiet=True)
    core.mdp.render(save_path=save_path+f"Epoch_0")

    core.learn(
        n_steps=initial_replay_size,
        n_steps_per_fit_per_agent=[initial_replay_size, initial_replay_size],
        quiet=True
    )

    epochbar = tqdm(range(n_epochs), unit='epoch', leave=False)
    for e in epochbar:
        core.learn(n_steps=n_steps_learn, n_steps_per_fit_per_agent=[1, 1], quiet=True)

        data.append(compute_metrics(core.evaluate(n_steps=n_steps_eval, render=False, quiet=True)))

        core.evaluate(n_steps=n_steps_eval, render=False, quiet=True)
        core.mdp.render(save_path=save_path+f"Epoch_{e + 1}")

    return np.array(data)


tuning_params1 = ["xxx"]
tuning_params2 = ["xxx"]

data = grid_search(
    tuning_params1=tuning_params1,
    tuning_params2=tuning_params2,
    seeds=[seed],
    train=train,
    base_path="Plots/"
)

plot_data(
    tuning_params1=tuning_params1,
    tuning_params2=tuning_params2,
    seeds=[seed],
    data=data,
    only_xy_plot=True
)
