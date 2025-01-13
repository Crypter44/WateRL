import time

from mushroom_rl.environments import DMControl
from tqdm import tqdm

from Mushroom.agents.ddpg import create_ddpg_agent, run_ddpg_training
from Mushroom.core.better_mujoco_core import BetterMujocoCore
from Mushroom.utils.utils import plot
from Mushroom.utils.utils import set_seed

# Parametrization
seeds = [1234, 2345]

horizon = 500
gamma = 0.99
gamma_eval = 1.

lr_actor = 1e-3
lr_critic = 5e-4

initial_replay_size = 500
max_replay_size = 5000
batch_size = 200

n_features = 80
tau = .001
sigma = 0.3
target_sigma = 0.001
theta = 0.15
dt = 1e-2

n_epochs = 140
n_steps_learn = 1000
n_steps_test = 2000
n_steps_per_fit = 1

disable_noise_for_evaluation = True

# Tuning parameters
tuning_params1 = ["xxx"]
tuning_params2 = ["xxx"]

# Save
save = True
save_every = 140

# create a dictionary to store data for each seed

t = time.time()

data = {}

pbar = tqdm(total=len(tuning_params1) * len(tuning_params2), unit="experiment", leave=False)
for p1 in tuning_params1:
    for p2 in tuning_params2:
        data[f"{p1}-{p2}"] = {}
        for seed in tqdm(total=len(seeds), unit="seed", leave=False, iterable=seeds):
            set_seed(seed)
            # MDP
            mdp = DMControl(domain_name='walker', task_name='stand', horizon=horizon, gamma=gamma)
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
            )

            # Core
            core = BetterMujocoCore(agent, mdp, custom_rendering_enabled=False)

            # Run training
            current_data = run_ddpg_training(
                core,
                n_epochs,
                n_steps_learn,
                n_steps_test,
                n_steps_per_fit,
                initial_replay_size,
                sigma=sigma,
                target_sigma=target_sigma,
                gamma_eval=gamma_eval,
                disable_noise_for_evaluation=disable_noise_for_evaluation,
                save=save,
                save_every=save_every,
                save_postfix=f"{p1}-{p2}-{seed}"
            )

            # Store the results
            data[f"{p1}-{p2}"][seed] = current_data
        pbar.update()

pbar.close()
print(f"Training took {time.time() - t:.2f} seconds")

# Plot the results
plot(tuning_params1, tuning_params2, seeds, data)  # TODO fix plotting
