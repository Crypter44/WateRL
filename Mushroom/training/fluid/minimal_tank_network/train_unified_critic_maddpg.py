import json

import matplotlib as mpl
import numpy as np
from tqdm import tqdm

from Mushroom.agents.agent_factory import setup_maddpg_agents_with_unified_critic
from Mushroom.agents.sigma_decay_policies import set_noise_for_all, update_sigma_for_all, UnivariateGaussianPolicy
from Mushroom.core.multi_agent_core_mixer import MultiAgentCoreMixer
from Mushroom.environments.fluid.minimal_tank_network import MinimalTankNetwork
from Mushroom.utils.plotting import plot_training_data, plot_debug_data
from Mushroom.utils.utils import set_seed, parametrized_training, compute_metrics_with_labeled_dataset, final_evaluation

# PARAMS
num_agents = 2
gamma = 0.99
gamma_eval = 1.

lr_actor = 2e-5
actor_activation = 'sigmoid'
lr_critic = 2e-5

initial_replay_size_episodes = 5
max_replay_size_episodes = 5000
batch_size = 200

n_features = 80
tau = .001
grad_norm_clipping = 0.5

sigma = [(0, 0.7), (10, 0.2)]
decay_type = 'exponential'
cut_of_exploration_when_converged = False

n_epochs = 40
n_episodes_learn = 10
n_episodes_test = 3
n_steps_per_fit = 1

n_episodes_final = 3
n_episodes_final_render = 1
n_epochs_per_checkpoint = 5

criteria = {
    "demand": {
        "w": 1,
        "max": 0,
        "min": -2,
        "value_at_bound": 0.001,
        "bound": 0.4
    },
}
demand_curve = "tagesgang"
# END_PARAMS

mpl.rcParams['figure.max_open_warning'] = n_episodes_final_render

# create a dictionary to store data for each seed
def train(p1, p2, seed, save_path):
    set_seed(seed)
    # MDP
    mdp = MinimalTankNetwork(gamma=p1, criteria=criteria, labeled_step=True, demand_curve=demand_curve)
    agents, maddpg = setup_maddpg_agents_with_unified_critic(
        mdp,
        policy=UnivariateGaussianPolicy(
            sigma_checkpoints=sigma,
            decay_type=decay_type,
        ),
        n_agents=num_agents,
        n_features_actor=n_features,
        lr_actor=lr_actor * p2,
        actor_activation=actor_activation,
        n_features_critic=n_features,
        lr_critic=lr_critic * p2,
        batch_size=batch_size,
        initial_replay_size=initial_replay_size_episodes * mdp.info.horizon,
        max_replay_size=max_replay_size_episodes * mdp.info.horizon,
        tau=tau,
        grad_norm_clip=grad_norm_clipping,
    )

    # Core
    core = MultiAgentCoreMixer(
        agents=agents,
        mixer=maddpg,
        mdp=mdp,
    )
    maddpg.set_debug_logging(True)
    debug_info = None

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
        debug_info = maddpg.get_debug_info(debug_info)

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
    plot_debug_data(debug_info, save_path)
    final_evaluation(n_episodes_final, n_episodes_final_render, core, save_path)

    return {"metrics": np.array(data), "additional_data": {}}


training_data, path = parametrized_training(
    __file__,
    [1, 10, 50],
    [0.99, 0.999],
    [1],
    train=train,
    base_path="Plots/MADDPG_unified_critic/",
)

plot_training_data(training_data, path)
