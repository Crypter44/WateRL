import json

import matplotlib as mpl
import numpy as np
from tqdm import tqdm

from Mushroom.agents.agent_factory import setup_facmac_agents
from Mushroom.agents.sigma_decay_policies import set_noise_for_all, update_sigma_for_all, UnivariateGaussianPolicy
from Mushroom.core.multi_agent_core_mixer import MultiAgentCoreMixer
from Mushroom.environments.fluid.minimal_tank_network import MinimalTankNetwork
from Mushroom.utils.plotting import plot_training_data
from Mushroom.utils.utils import set_seed, parametrized_training, compute_metrics_with_labeled_dataset

# PARAMS
num_agents = 2
gamma = 0.99
gamma_eval = 1.

lr_actor = 2e-5
lr_critic = 2e-5

initial_replay_size_episodes = 5
max_replay_size_episodes = 5000
batch_size = 200

n_features = 80
tau = .005
grad_norm_clipping = 0.5

sigma = [(0, 0.6), (30, 0.1), (50, 0.01)]
decay_type = 'exponential'

n_epochs = 30
n_episodes_learn = 10
n_episodes_test = 3
n_steps_per_fit = 1

n_episodes_final = 3
n_episodes_final_render = 1
n_epochs_per_checkpoint = 5

criteria = {
    "demand_switch": {
        "w": 1,
        "tolerance": 0.025,
    },
}

demand_curve = "tagesgang"
# END_PARAMS

mpl.rcParams['figure.max_open_warning'] = n_episodes_final_render


# create a dictionary to store data for each seed
def train(p1, p2, seed, save_path):
    set_seed(seed)
    # MDP
    mdp = MinimalTankNetwork(criteria=criteria, labeled_step=True, demand_curve=demand_curve)
    agents, facmac = setup_facmac_agents(
        mdp,
        policy=None,
        n_agents=num_agents,
        n_features_actor=n_features,
        lr_actor=lr_actor,
        n_features_critic=n_features,
        lr_critic=lr_critic,
        batch_size=batch_size,
        initial_replay_size=initial_replay_size_episodes * mdp.info.horizon,
        max_replay_size=max_replay_size_episodes * mdp.info.horizon,
        tau=tau,
        grad_norm_clip=grad_norm_clipping,
        sigma_checkpoints=sigma,
    )

    # Core
    core = MultiAgentCoreMixer(
        agents=agents,
        mixer=facmac,
        mdp=mdp,
    )

    core.learn(
        n_steps=initial_replay_size_episodes * mdp.info.horizon,
        n_steps_per_fit_per_agent=[initial_replay_size_episodes * mdp.info.horizon] * num_agents,
    )
    set_noise_for_all(core.agents, False)
    data = [compute_metrics_with_labeled_dataset(core.evaluate(n_episodes=n_episodes_test, render=False)[0])]
    set_noise_for_all(core.agents, True)
    sigma_list = [core.agents[0].policy.get_sigma()]
    core.mdp.render(save_path=save_path + f"Epoch_0")

    pbar = tqdm(range(n_epochs), unit='epoch', leave=False)
    for n in pbar:
        core.learn(
            n_episodes=n_episodes_learn,
            n_steps_per_fit_per_agent=[n_steps_per_fit] * num_agents,
        )

        core.evaluate(n_episodes=1, render=False, quiet=True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}_Noisy")
        set_noise_for_all(core.agents, False)
        dataset, _ = core.evaluate(n_episodes=n_episodes_test, render=False, )
        set_noise_for_all(core.agents, True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}")

        data.append(compute_metrics_with_labeled_dataset(dataset))
        if data[-1][2] >= 280:
            update_sigma_for_all(core.agents, 0.005)
        else:
            update_sigma_for_all(core.agents)
        sigma_list.append(core.agents[0].policy.get_sigma())

        pbar.set_postfix(
            MinMaxMean=np.round(data[-1][0:3], 2),
            sigma=np.round(core.agents[0].policy.get_sigma(), 2)
        )

        if (n + 1) % n_epochs_per_checkpoint == 0:
            for i, a in enumerate(core.agents):
                a.save(save_path + f"/checkpoints/Epoch_{n + 1}_Agent_{i}")

    set_noise_for_all(core.agents, False)
    for i in range(n_episodes_final_render):
        core.evaluate(n_episodes=1, quiet=True)
        core.mdp.render(save_path=save_path + f"Final_{i}")

    for i, a in enumerate(core.agents):
        a.save(save_path + f"/checkpoints/Final_Agent_{i}")

    if n_episodes_final > 0:
        with open(save_path + "Evaluation.json", "w") as f:
            final = compute_metrics_with_labeled_dataset(
                core.evaluate(n_episodes=n_episodes_final, render=False)[0],
                gamma_eval
            )
            json.dump({
                "Min": final[0],
                "Max": final[1],
                "Mean": final[2],
                "Median": final[3],
                "Count": final[4],
            }, f, indent=4)

    return {
        "metrics": np.array(data),
        "additional_data": {
            "sigma": sigma_list,
        }}


training_data, path = parametrized_training(
    __file__,
    # [0.01, 0.05, 0.1, 0.5, 1],
    # [0.1, 0.2, 0.5, 1, 2, 5, 10],
    [None],
    [None],
    [1],
    train=train,
    base_path="Plots/FACMAC/",
    save_whole_file=True,
)

plot_training_data(training_data, path, True)
