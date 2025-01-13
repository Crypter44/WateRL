import numpy as np
from mushroom_rl.utils.dataset import compute_metrics
from tqdm import tqdm

from Mushroom.agents.ddpg import create_ddpg_agent
from Mushroom.agents.sigma_decay_policies import set_noise_for_all, update_sigma_for_all
from Mushroom.environments.fluid.circular_network_no_pi import CircularFluidNetworkWithoutPI
from Mushroom.core.multi_agent_core import MultiAgentCore
from Mushroom.utils.plotting import plot_training_data
from Mushroom.utils.utils import set_seed, parametrized_training

# PARAMS
gamma = 0.99
gamma_eval = 1.

lr_actor = 5e-5
lr_critic = 1e-4

initial_replay_size = 500
max_replay_size = 5000
batch_size = 200

n_features = 80
tau = .005

sigma = 0.5
target_sigma = 0.005
sigma_transition_length = 30

theta = 0.15
dt = 1e-2

n_epochs = 40
n_steps_learn = 1400
n_steps_test = 600
n_steps_per_fit = 1

num_agents = 4
power_penalty = 0.0

criteria = {
    "demand": 1.0,
}

# END_PARAMS


# create a dictionary to store data for each seed
def train(p1, p2, seed, save_path):
    set_seed(seed)
    # MDP
    mdp = CircularFluidNetworkWithoutPI(gamma=gamma, criteria=criteria)
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
        sigma=sigma,
        target_sigma=target_sigma,
        sigma_transition_length=sigma_transition_length,
        theta=theta,
        dt=dt,
    ) for i in range(num_agents)]

    # Core
    core = MultiAgentCore(agent=agents, mdp=mdp)

    core.learn(
        n_steps=initial_replay_size,
        n_steps_per_fit_per_agent=[initial_replay_size]*num_agents,
        quiet=True
    )
    data = [compute_metrics(core.evaluate(n_steps=n_steps_test, render=False, quiet=True), gamma_eval)]
    core.mdp.render(save_path=save_path + f"Epoch_0")

    pbar = tqdm(range(n_epochs), unit='epoch', leave=False)
    for n in pbar:
        core.learn(
            n_steps=n_steps_learn,
            n_steps_per_fit_per_agent=[n_steps_per_fit]*num_agents,
            quiet=True
        )

        core.evaluate(n_steps=200, render=False, quiet=True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}_Noisy")
        set_noise_for_all(core.agent, False)
        dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
        set_noise_for_all(core.agent, True)
        core.mdp.render(save_path=save_path + f"Epoch_{n+1}")

        data.append(compute_metrics(dataset, gamma_eval))
        pbar.set_postfix(
            MinMaxMean=np.round(data[-1][0:3], 2),
            sigma=np.round(core.agent[0].policy._sigma, 2)
        )

        update_sigma_for_all(agents)

    set_noise_for_all(core.agent, False)
    for i in range(50):
        core.evaluate(n_episodes=1, quiet=True)
        core.mdp.render(save_path=save_path + f"Final_{i}")

    return {"metrics": np.array(data), "additional_data": {}}


training_data, path = parametrized_training(
    __file__,
    [None],
    [None],
    [0],
    train=train,
    base_path="./Plots/DDPG/",
)

plot_training_data(training_data, path)
