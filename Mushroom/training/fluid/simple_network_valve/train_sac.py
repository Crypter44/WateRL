from mushroom_rl.core import Core
from tqdm import tqdm

from Mushroom.agents.sac import create_sac_agent, run_sac_training
from Mushroom.environments.fluid.simple_network_valve import SimpleNetworkValve
from Mushroom.utils.utils import plot
from Mushroom.utils.utils import set_seed

# Parametrization
seeds = [3, 33]

horizon = 100
gamma = 0.99
gamma_eval = 1.

lr_actor = 1e-3
lr_critic = 6e-4
lr_alpha = 3e-3

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
n_recordings = 1

# Tuning
tuning_params1 = ["xxx"]
tuning_params2 = ["xxx"]

# Training
data = {}
pbar = tqdm(total=len(tuning_params1) * len(tuning_params2), unit='experiment')
for p1 in tuning_params1:
    for p2 in tuning_params2:
        seedbar = tqdm(total=len(seeds), unit='seed', leave=False)
        current_data = {}
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
                record_postfix=f" - s:{seed} - p1:{p1} - p2:{p2}",
                n_recordings=n_recordings
            )

            current_data[seed] = results
            seedbar.update(1)
        seedbar.close()

        data[f'{p1}-{p2}'] = current_data
        pbar.update(1)

plot(tuning_params1, tuning_params2, seeds, data)
