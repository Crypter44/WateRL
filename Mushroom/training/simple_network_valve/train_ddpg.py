from mushroom_rl.core import Core

from Mushroom.agents.ddpg import create_ddpg_agent, run_ddpg_training
from Mushroom.fluid_network_environments.simple_network_valve import SimpleNetworkValve
from Mushroom.utils import plot
from Mushroom.utils import set_seed

# Parametrization
seeds = [4403, 2003, 2024, 5264]

horizon = 100
gamma = 0.99
gamma_eval = 1.

lr_actor = 1e-4
lr_critic = 1e-3

initial_replay_size = 500
max_replay_size = 5000
batch_size = 200

n_features = 5
tau = .001
sigma = 0.08
theta = 0.15
dt = 1e-2

n_epochs = 60
n_steps_learn = 600
n_steps_test = 400
n_steps_per_fit = 1

disable_noise_for_evaluation = True

# Tuning parameters
tuning_params1 = ["N/A"]
tuning_params2 = ["N/A"]

# Record
record = False
record_every = 15

# create a dictionary to store data for each seed
data = {}
for p1 in tuning_params1:
    for p2 in tuning_params2:
        data[f"{p1}-{p2}"] = {}
        for seed in seeds:
            set_seed(seed)
            # MDP
            mdp = SimpleNetworkValve(gamma, horizon)
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
                theta=theta,
                dt=dt,
            )

            # Core
            core = Core(agent, mdp)

            # Run training
            current_data = run_ddpg_training(
                core,
                n_epochs,
                n_steps_learn,
                n_steps_test,
                n_steps_per_fit,
                initial_replay_size,
                gamma_eval,
                disable_noise_for_evaluation=disable_noise_for_evaluation,
                save=record,
                save_every=record_every,
                save_postfix=f"{p1}-{p2}-{seed}"
            )

            # Store the results
            data[f"{p1}-{p2}"][seed] = current_data

# Plot the results
plot(tuning_params1, tuning_params2, seeds, data)
