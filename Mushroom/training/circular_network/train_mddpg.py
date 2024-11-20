import numpy as np
from mushroom_rl.utils.dataset import compute_metrics
from tqdm import tqdm

from Mushroom.agents.ddpg import create_ddpg_agent
from Mushroom.fluid_network_environments.circular_network import CircularFluidNetwork
from Mushroom.multi_agent_core import MultiAgentCore
from Mushroom.plotting import plot_training_data
from Mushroom.utils import set_seed, grid_search

gamma = 0.99
gamma_eval = 1.

lr_actor = 1e-4
lr_critic = 6e-4

initial_replay_size = 500
max_replay_size = 5000
batch_size = 200

n_features = 80
tau = .005
sigma = 0.3
target_sigma = 0.15
theta = 0.15
dt = 1e-2

n_epochs = 30
n_steps_learn = 400
n_steps_test = 600
n_steps_per_fit = 1


# create a dictionary to store data for each seed
def train(p1, p2, seed, save_path):
    set_seed(seed)
    # MDP
    mdp = CircularFluidNetwork(gamma=gamma, power_penalty=p1)
    agents = [create_ddpg_agent(
        mdp,
        n_features_actor=n_features,
        lr_actor=lr_actor,
        n_features_critic=n_features,
        lr_critic=lr_critic,
        batch_size=batch_size,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        tau=tau,
        sigma=sigma,
        theta=theta,
        dt=dt,
    ) for _ in range(2)]

    # Core
    core = MultiAgentCore(agent=agents, mdp=mdp)

    core.learn(
        n_steps=initial_replay_size,
        n_steps_per_fit_per_agent=[initial_replay_size, initial_replay_size],
        quiet=True
    )
    data = [compute_metrics(core.evaluate(n_steps=n_steps_test, render=False, quiet=True), gamma_eval)]
    core.mdp.render(save_path=save_path + f"Epoch_0")

    pbar = tqdm(range(n_epochs), unit='epoch', leave=False)
    for n in pbar:
        core.learn(
            n_steps=n_steps_learn,
            n_steps_per_fit_per_agent=[n_steps_per_fit, n_steps_per_fit],
            quiet=True
        )

        sigmas = [core.agent[0].policy._sigma, core.agent[1].policy._sigma]
        core.agent[0].policy._sigma = 0
        core.agent[1].policy._sigma = 0
        dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
        core.agent[0].policy._sigma = sigmas[0]
        core.agent[1].policy._sigma = sigmas[1]
        core.mdp.render(save_path=save_path + f"Epoch_{n+1}")

        data.append(compute_metrics(dataset, gamma_eval))
        pbar.set_postfix(
            MinMaxMean=np.round(data[-1][0:3], 2),
            sigma=np.round(core.agent[0].policy._sigma, 2)
        )

        core.agent[0].policy._sigma *= (target_sigma / sigma) ** (1 / n_epochs)
        core.agent[1].policy._sigma *= (target_sigma / sigma) ** (1 / n_epochs)

    sigmas = [core.agent[0].policy._sigma, core.agent[1].policy._sigma]
    core.agent[0].policy._sigma = 0
    core.agent[1].policy._sigma = 0
    for i in range(50):
        core.evaluate(n_episodes=1, quiet=True)
        core.mdp.render(save_path=save_path + f"Final_{i}")

    return {"metrics": np.array(data), "additional_data": {}}


training_data, path = grid_search(
    [0.05, 0.1],
    [None],
    [0],
    train,
    "./Plots/DDPG/"
)

plot_training_data(training_data, path)
