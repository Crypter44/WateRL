# PARAMS
import numpy as np
from matplotlib import pyplot as plt
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_metrics
from tqdm import tqdm

from Mushroom.agents.sac import create_sac_agent
from Mushroom.utils.plotting import plot_training_data
from Mushroom.environments.test.constant_value_env import ConstantValueEnv
from Mushroom.utils.utils import set_seed, parametrized_training

# PARAMS
n_features_actor = 80
lr_actor = 1e-4
n_features_critic = 80
lr_critic = 2e-4

batch_size = 200
initial_replay_size = 500
max_replay_size = 10000

warmup_transitions = initial_replay_size
tau = 0.005

lr_alpha = 1e-4
log_std_min = -20
log_std_max = np.log(3)
target_entropy = -5

n_epochs = 60
n_steps_learn = 2000
n_steps_eval = 600


def reward_fn(action):
    # expected value 0.896575
    f = np.sin(10 * action[0] ** 2) * action[0] ** 3 + -0.00848
    r = np.exp(-4 * (f - value) ** 2)
    return r


value = .7
start_state = np.array([0] * 4)
steps_until_state_change = 12
reset_to_start = True

# END_PARAMS


def train(p1, p2, seed, save_path):
    sigma_all = []
    mu_all = []
    temp = []
    entropy = []

    set_seed(seed)
    # MDP
    mdp = ConstantValueEnv(
        value=value,
        start_state=start_state,
        steps_until_state_change=steps_until_state_change,
        reset_to_start=reset_to_start,
        reward_fn=reward_fn,
    )
    agents, mu, sigma = create_sac_agent(
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
    )

    # Core
    core = Core(agent=agents, mdp=mdp)

    core.learn(
        n_steps=initial_replay_size,
        n_steps_per_fit=initial_replay_size,
        quiet=True
    )
    data = [compute_metrics(core.evaluate(n_steps=n_steps_eval, render=False, quiet=True), 1.0)]
    core.mdp.render(save_path=save_path + f"Epoch_0")
    plt.plot([m for m in mu[-200:]], label='mu')
    plt.title(f"Mu - Epoch 0")
    plt.savefig(save_path + f"Mu_Epoch_0")
    plt.clf()
    plt.plot([s for s in sigma[-200:]], label='sigma')
    plt.title(f"Sigma - Epoch 0")
    plt.savefig(save_path + f"Sigma_Epoch_0")
    plt.clf()

    pbar = tqdm(range(n_epochs), unit='epoch', leave=False)
    for n in pbar:
        core.learn(
            n_steps=n_steps_learn,
            n_steps_per_fit=1,
            quiet=True
        )

        dataset = core.evaluate(n_steps=n_steps_eval, render=False, quiet=True)
        if (n + 1) % 1 == 0:
            core.mdp.render(save_path=save_path + f"Epoch_{n + 1}")
            plt.plot([m for m in mu[-200:]], label='mu')
            plt.legend()
            plt.title(f"Mu - Epoch {n + 1}")
            plt.savefig(save_path + f"Mu_Epoch_{n + 1}")
            plt.clf()
            plt.plot([s for s in sigma[-200:]], label='sigma')
            plt.title(f"Sigma - Epoch {n + 1}")
            plt.savefig(save_path + f"Sigma_Epoch_{n + 1}")
            plt.clf()

        data.append(compute_metrics(dataset, 1.0))
        pbar.set_postfix(
            MinMaxMean=np.round(data[-1][0:3], 2),
        )
        sigma_all.append(np.mean(sigma[-200:]))
        mu_all.append(np.mean(mu[-200:]))
        temp.append(core.agent._alpha_np)
        entropy.append(core.agent.policy.entropy(core.mdp._state))

    for i in range(10):
        core.evaluate(n_episodes=1, quiet=True)
        core.mdp.render(save_path=save_path + f"Final_{i}")

    return {
        "metrics": np.array(data[:]),
        "additional_data": {
            "sigma": np.array(sigma_all),
            "mu": np.array(mu_all),
            "alpha": np.array(temp),
            "entropy": np.array(entropy),
        }
    }


training_data, path = parametrized_training(
    __file__,
    [None],
    [None],
    [0],
    train=train,
    base_path="./Plots/SAC/"
)

plot_training_data(training_data, path, plot_additional_data=True)
