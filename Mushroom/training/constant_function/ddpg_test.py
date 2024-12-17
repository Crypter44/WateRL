import numpy as np
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_metrics
from tqdm import tqdm

from Mushroom.agents.ddpg import create_ddpg_agent
from Mushroom.agents.sigma_decay_policies import set_noise_for_all, update_sigma_for_all
from Mushroom.plotting import plot_training_data
from Mushroom.training.constant_function.constant_value_env import ConstantValueEnv
from Mushroom.utils import grid_search, set_seed

# PARAMS
gamma = 0.99
gamma_eval = 1.

lr_actor = 5e-4
lr_critic = 1e-3

initial_replay_size = 500
max_replay_size = 5000
batch_size = 200

n_features = 80
tau = .005

sigma = 0.5
target_sigma = 0.2
sigma_transition_length = 30

theta = 0.15
dt = 1e-2

n_epochs = 30
n_steps_learn = 1400
n_steps_test = 600
n_steps_per_fit = 1
num_agents = 1


def reward_fn(action):
    f = np.sin(10 * action[0] ** 2) * action[0] ** 3 + -0.00848
    # r = -np.abs(f - value) * 10
    r = np.exp(-4 * (f - value) ** 2)
    return r


value = .7
start_state = np.array([0] * 4)
steps_until_state_change = 12
reset_to_start_state = True


# END_PARAMS


def train(p1, p2, seed, save_path):
    set_seed(seed)
    mdp = ConstantValueEnv(
        value,
        reward_fn=reward_fn,
        start_state=start_state,
        steps_until_state_change=steps_until_state_change,
        reset_to_start=reset_to_start_state,
    )
    agent = create_ddpg_agent(
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
        target_sigma=target_sigma,
        sigma_transition_length=sigma_transition_length,
        theta=theta,
        dt=dt,
    )
    core = Core(agent=agent, mdp=mdp)

    core.learn(
        n_steps=initial_replay_size,
        n_steps_per_fit=initial_replay_size,
        quiet=True
    )
    data = [compute_metrics(core.evaluate(n_steps=n_steps_test, render=False, quiet=True), gamma_eval)]
    core.mdp.render(save_path=save_path + f"Epoch_0")

    pbar = tqdm(range(n_epochs), unit='epoch', leave=False)
    for n in pbar:
        core.learn(n_steps=n_steps_learn, n_steps_per_fit=n_steps_per_fit, quiet=True)

        dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}_Noise")
        set_noise_for_all(core.agent, False)
        dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}")
        data.append(compute_metrics(dataset, gamma_eval))
        set_noise_for_all(core.agent, True)
        update_sigma_for_all(core.agent)

    set_noise_for_all(core.agent, False)
    for i in range(1):
        core.evaluate(n_episodes=1, quiet=True)
        core.mdp.render(save_path=save_path + f"Final_{i}")

    return {"metrics": np.array(data[:]), "additional_data": {}}


training_data, path = grid_search(
    __file__,
    [None],
    [None],
    [0],
    train=train,
    base_path="./Plots/DDPG/"
)

plot_training_data(training_data, path)
