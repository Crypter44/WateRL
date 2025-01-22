import numpy as np
from mushroom_rl.utils.dataset import compute_metrics
from tqdm import tqdm

from Mushroom.agents.sac import create_sac_agent
from Mushroom.environments.fluid.circular_network_no_pi import CircularFluidNetworkWithoutPI
from Mushroom.core.multi_agent_core import MultiAgentCore
from Mushroom.utils.plotting import plot_training_data
from Mushroom.utils.utils import set_seed, parametrized_training

# PARAMS
seed = 0

n_features_actor = 80
lr_actor = 2e-6
n_features_critic = 80
lr_critic = 4e-6
batch_size = 200
initial_replay_size = 500
max_replay_size = 10000
warmup_transitions = 0
tau = 0.005
lr_alpha = 0.001
log_std_min = -20
log_std_max = np.log(1)
target_entropy = -5

n_epochs = 30
n_steps_learn = 400
n_steps_eval = 600
renders_on_completion = 50

criteria = {
    "demand": {"w": 1.0},
}
# END_PARAMS


def train(p1, p2, seed, save_path):
    set_seed(seed)

    mdp = CircularFluidNetworkWithoutPI(gamma=0.99, criteria=criteria)
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
        )[0] for _ in range(6)
    ]

    core = MultiAgentCore(agent=agents, mdp=mdp)

    data = [compute_metrics(core.evaluate(n_steps=n_steps_eval, render=False, quiet=True))]
    temp = [[core.agent[0]._alpha_np], [core.agent[1]._alpha_np]]
    entropy = [
        [core.agent[0].policy.entropy(core.mdp._current_state)],
        [core.agent[1].policy.entropy(core.mdp._current_state)]
    ]

    core.evaluate(n_steps=n_steps_eval, render=False, quiet=True)
    core.mdp.render(save_path=save_path + f"Epoch_0")

    core.learn(
        n_steps=initial_replay_size,
        n_steps_per_fit_per_agent=[initial_replay_size] * 6,
        quiet=True
    )

    epochbar = tqdm(range(n_epochs), unit='epoch', leave=False)
    for e in epochbar:
        core.learn(n_steps=n_steps_learn, n_steps_per_fit_per_agent=[1] * 6, quiet=True)

        data.append(compute_metrics(core.evaluate(n_steps=n_steps_eval, render=False, quiet=True)))

        core.evaluate(n_steps=n_steps_eval, render=False, quiet=True)
        core.mdp.render(save_path=save_path + f"Epoch_{e + 1}")

        temp[0].append(core.agent[0]._alpha_np)
        temp[1].append(core.agent[1]._alpha_np)
        entropy[0].append(core.agent[0].policy.entropy(core.mdp._current_state))
        entropy[1].append(core.agent[1].policy.entropy(core.mdp._current_state))

    for i in range(renders_on_completion):
        core.evaluate(n_episodes=1, quiet=True)
        core.mdp.render(save_path=save_path + f"Final_{i}")

    return {
        "metrics": np.array(data),
        "additional_data": {
            "alpha 0": np.array(temp[0]),
            "alpha 1": np.array(temp[1]),
            "entropy 0": np.array(entropy[0]),
            "entropy 1": np.array(entropy[1]),
        }
    }


tuning_params1 = [None]
tuning_params2 = [None]

data, path = parametrized_training(
    __file__,
    tuning_params1=tuning_params1,
    tuning_params2=tuning_params2,
    seeds=[seed],
    train=train,
    base_path="Plots/SAC/"
)

plot_training_data(data, path, plot_additional_data=True)
