import numpy as np
from matplotlib import pyplot as plt
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_metrics
from tqdm import tqdm

from Mushroom.ddpg_agent import create_ddpg_agent
from Mushroom.fluid_network_environments.simple_network_valve import SimpleNetworkValve
from Mushroom.utils import plot_to_ax
from utils import set_seed, plot_multiple_seeds

# Parametrization
seeds = [526, 42, 7, 999, 1]

horizon = 50
gamma = 0.99
gamma_eval = 1.

lr_actor = 1e-4
lr_critic = 1e-3

initial_replay_size = 500
max_replay_size = 5000
batch_size = 200
n_features = 5
tau = .001
sigma = 0.2
theta = 0.15
dt = 1e-2

n_epochs = 50
n_steps_learn = 200
n_steps_test = 400
n_steps_per_fit = 1

# Tuning parameters
tuning_params1 = [1e-4]
tuning_params2 = [1e-3]

def run_training(
        core,
        n_epochs,
        n_steps_learn,
        n_steps_test,
        n_steps_per_fit,
        initial_replay_size,
        gamma_eval,
        record=False,
        record_name="rl-experiment",
        record_every=5,
        record_dir="./Videos"
):
    # Fill the replay memory with random samples
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, quiet=True)

    # ------------- RUN EXPERIMENT ------------- #
    dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
    data = []

    metrics = compute_metrics(dataset, gamma_eval)
    data.append(metrics)

    pbar = tqdm(range(n_epochs), desc='Running... ', unit='epoch')
    for n in pbar:
        # learning step, agent learns from 1000 steps -> 2 episodes
        core.learn(n_steps=n_steps_learn, n_steps_per_fit=n_steps_per_fit, quiet=True)
        # evaluation step, agent evaluates 2000 steps -> 4 episodes
        dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
        # if record and (n + 1) % record_every == 0:
        #     core.render_episode(f"{record_dir}/{record_name}-ep{n + 1}.mp4")
        metrics = compute_metrics(dataset, gamma_eval)
        data.append(metrics)
        pbar.set_postfix(Metrics=metrics)

    # core.clear_render_cache()

    return np.array(data)

# create a dictionary to store data for each seed
data = {}
for p1 in tuning_params1:
    for p2 in tuning_params2:
        data[f"{p1}-{p2}"] = {}
        for seed in seeds:
            set_seed(seed)
            # MDP
            mdp = SimpleNetworkValve(gamma, horizon)

            # Algorithm
            core = Core(
                create_ddpg_agent(
                    mdp,
                    n_features_actor=n_features,
                    lr_actor=p1,
                    n_features_critic=n_features,
                    lr_critic=p2,
                    batch_size=batch_size,
                    initial_replay_size=initial_replay_size,
                    max_replay_size=max_replay_size,
                    tau=tau,
                    sigma=sigma,
                    theta=theta,
                    dt=dt,
                ), mdp
            )

            # Run training
            current_data = run_training(
                core,
                n_epochs,
                n_steps_learn,
                n_steps_test,
                n_steps_per_fit,
                initial_replay_size,
                gamma_eval,
                record=True,
                record_name=f"walker-stand-{seed}-p1-{p1}-p2-{p2}",
            )

            # Store the results
            data[f"{p1}-{p2}"][seed] = current_data

# Plot the results
fig, ax = plt.subplots(len(tuning_params1), len(tuning_params2), figsize=(len(tuning_params2) * 8, len(tuning_params1) * 8))
x = 0
y = 0
for p1 in tuning_params1:
    y = 0
    for p2 in tuning_params2:
        fig1, ax1 = plot_multiple_seeds(data[f"{p1}-{p2}"], f"Walker Stand p1={p1} p2={p2}", True)
        fig2, ax2 = plot_multiple_seeds(data[f"{p1}-{p2}"], f"Walker Stand p1={p1} p2={p2}", False)

        fig1.show()
        fig2.show()

        plot_to_ax(ax[x, y], data[f"{p1}-{p2}"], f"Walker Stand p1={p1} p2={p2}", 0.1)
        y += 1
    x += 1

fig.show()
