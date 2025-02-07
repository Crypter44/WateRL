import numpy as np
from tqdm import tqdm

from Mushroom.agents.agent_factory import setup_facmac_agents
from Mushroom.agents.sigma_decay_policies import set_noise_for_all, update_sigma_for_all, UnivariateGaussianPolicy
from Mushroom.core.multi_agent_core_mixer import MultiAgentCoreMixer
from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Mushroom.utils.plotting import plot_training_data
from Mushroom.utils.utils import set_seed, parametrized_training, compute_metrics_with_labeled_dataset

# PARAMS
gamma = 0.99
gamma_eval = 1.

lr_actor = 1e-5
lr_critic = 1e-5

initial_replay_size = 500
max_replay_size = 5000
batch_size = 200

n_features = 80
tau = .005

sigma_checkpoints = [(0, 0.4), (50, 0.15), (75, 0.05)]
decay_type = 'linear'

n_epochs = 20
n_steps_learn = 1000
n_steps_test = 300
n_steps_per_fit = 1

num_agents = 2
criteria = {
    "target_speed": {
        "w": 1.,
        "target": 0.5
    },
}
demand = ("constant", 0.5, 0.5)
# END_PARAMS


# create a dictionary to store data for each seed
def train(p1, p2, seed, save_path):

    criteria["target_speed"]["target"] = p1
    base_mul, critic_mul = p2

    set_seed(seed)
    mdp = CircularFluidNetwork(gamma=gamma, criteria=criteria, labeled_step=True, demand=demand)
    agents, facmac = setup_facmac_agents(
        mdp,
        policy=UnivariateGaussianPolicy(
            sigma_checkpoints=sigma_checkpoints,
            decay_type=decay_type,
        ),
        n_agents=num_agents,
        n_features_actor=n_features,
        lr_actor=lr_actor * base_mul,
        n_features_critic=n_features,
        lr_critic=lr_actor * base_mul * critic_mul,
        batch_size=batch_size,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        tau=tau,
    )
    mdp.enable_q_logging(agents)

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

        core.evaluate(n_episodes=1, render=False, quiet=True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}_Noisy")
        set_noise_for_all(core.agents, False)
        dataset, _ = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
        set_noise_for_all(core.agents, True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}")

        data.append(compute_metrics_with_labeled_dataset(dataset, gamma_eval))
        pbar.set_postfix(
            MinMaxMean=np.round(data[-1][0:3], 2),
            sigma=np.round(core.agents[0].policy._sigma_decay.get(), 2)
        )

        update_sigma_for_all(agents)

    set_noise_for_all(core.agents, False)
    for i in range(0):
        core.evaluate(n_episodes=1, quiet=True)
        core.mdp.render(save_path=save_path + f"Final_{i}")

    mdp.stop_renderer()

    return {"metrics": np.array(data), "additional_data": {}}


training_data, path = parametrized_training(
    __file__,
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    [
        (1, 0.2), (1, 0.5), (1, 1), (1, 2), (1, 5),
        (2, 0.2), (2, 0.5), (2, 1), (2, 2), (2, 5),
    ],
    [1],
    train=train,
    base_path="./Plots/FACMAC/",
)

plot_training_data(training_data, path)
