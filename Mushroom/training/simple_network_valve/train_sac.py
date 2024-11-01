from mushroom_rl.core import Core

from Mushroom.agents.sac import create_sac_agent, run_sac_training
from Mushroom.fluid_network_environments.simple_network_valve import SimpleNetworkValve
from Mushroom.utils import plot_multiple_seeds
from Mushroom.utils import set_seed

# Parametrization
seeds = [1234, 2345]

horizon = 100
gamma = 0.99
gamma_eval = 1.

lr_actor = 1e-4
lr_critic = 1e-3
lr_alpha = 3e-4

initial_replay_size = 500
max_replay_size = 5000
batch_size = 200

n_features_actor = 80
n_features_critic = 80
tau = .005
log_std_min = -20
log_std_max = 2
target_entropy = -200.0

n_epochs = 60
n_steps_learn = 600
n_steps_test = 400
n_steps_per_fit = 1
warmup_transitions = initial_replay_size

# Record
record = True
record_every = 60
n_recordings = 4

# Training
data = {}
for seed in seeds:
    set_seed(seed)

    mdp = SimpleNetworkValve(gamma, horizon)

    agent = create_sac_agent(
        mdp,
        n_features_actor=n_features_actor,
        n_features_critic=n_features_critic,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        lr_alpha=lr_alpha,
        batch_size=batch_size,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        warmup_transitions=warmup_transitions,
        tau=tau,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        target_entropy=target_entropy
    )

    core = Core(agent, mdp)

    results = run_sac_training(
        core,
        n_epochs=n_epochs,
        n_steps_learn=n_steps_learn,
        n_steps_test=n_steps_test,
        n_steps_per_fit=n_steps_per_fit,
        initial_replay_size=initial_replay_size,
        gamma_eval=gamma_eval,
        record=record,
        record_every=record_every,
        record_postfix=f" - s:{seed}",
        n_recordings=n_recordings
    )

    data[seed] = results

# Plot results
fig, _ = plot_multiple_seeds(data, "SAC on Simple Network Valve")
fig2, _ = plot_multiple_seeds(data, "SAC on Simple Network Valve", False)
fig.show()
fig2.show()
