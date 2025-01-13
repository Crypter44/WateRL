import numpy as np
from tqdm import tqdm

from Mushroom.agents.facmac import setup_facmac_agents
from Mushroom.agents.sigma_decay_policies import set_noise_for_all, update_sigma_for_all
from Mushroom.core.multi_agent_core_mixer import MultiAgentCoreMixer
from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Mushroom.utils.plotting import plot_training_data
from Mushroom.utils.utils import set_seed, parametrized_training, compute_metrics_with_labeled_dataset

# PARAMS
gamma = 0.99
gamma_eval = 1.

lr_actor = 1e-4
lr_critic = 2e-4

initial_replay_size = 500
max_replay_size = 5000
batch_size = 200

n_features = 80
tau = .005

sigma = 0.5
target_sigma = 0.01
sigma_transition_length = 30

n_epochs = 30
n_steps_learn = 1000
n_steps_test = 600
n_steps_per_fit = 1

num_agents = 2
criteria = {
    "demand": 0.9,
    "max_power": 0.1,
    "negative_flow": 0.0
}
# END_PARAMS


# create a dictionary to store data for each seed
def train(p1, p2, seed, save_path):
    set_seed(seed)
    # MDP
    criteria["max_power"] = p1
    criteria["demand"] = 1.0 - p1
    mdp = CircularFluidNetwork(gamma=gamma, criteria=criteria, labeled_step=True)
    agents, facmac = setup_facmac_agents(
        mdp,
        n_agents=num_agents,
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
    )

    # Core
    core = MultiAgentCoreMixer(
        agents=agents,
        mixer=facmac,
        mdp=mdp,
    )

    core.learn(
        n_steps=initial_replay_size,
        n_steps_per_fit_per_agent=[initial_replay_size] * num_agents,
        quiet=True
    )
    dataset, _ = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
    data = [compute_metrics_with_labeled_dataset(dataset, gamma_eval)]
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
        set_noise_for_all(core.agents, False)
        dataset, _ = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
        set_noise_for_all(core.agents, True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}")

        data.append(compute_metrics_with_labeled_dataset(dataset, gamma_eval))
        pbar.set_postfix(
            MinMaxMean=np.round(data[-1][0:3], 2),
            sigma=np.round(core.agents[0].policy._sigma, 2)
        )

        update_sigma_for_all(agents)

    set_noise_for_all(core.agents, False)
    for i in range(50):
        core.evaluate(n_episodes=1, quiet=True)
        core.mdp.render(save_path=save_path + f"Final_{i}")

    return {"metrics": np.array(data), "additional_data": {}}


training_data, path = parametrized_training(
    __file__,
    [0.1, 0.25, 0.5],
    [None],
    [1],
    train=train,
    base_path="Plots/FACMAC/",
)

plot_training_data(training_data, path)
