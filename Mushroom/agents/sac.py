import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic.deep_actor_critic.sac import SAC, SACPolicy
from mushroom_rl.utils.dataset import compute_metrics
from torch import optim
from tqdm import tqdm


class ActorMuNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorMuNetwork, self).__init__()

        # Layers
        self.fc1 = nn.Linear(input_shape[-1], n_features)
        self.fc2 = nn.Linear(n_features, n_features)
        self.mu = nn.Linear(n_features, output_shape[0])

        # Initialization
        nn.init.xavier_uniform_(
            self.fc1.weight,
            gain=nn.init.calculate_gain('relu')
        )
        nn.init.xavier_uniform_(
            self.fc2.weight,
            gain=nn.init.calculate_gain('relu')
        )
        nn.init.xavier_uniform_(
            self.mu.weight,
            gain=nn.init.calculate_gain('tanh')
        )

    def forward(self, state):
        state = torch.squeeze(state, 1).float()
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # we choose linear again, since SAC already applies a tanh activation bounded to the action space
        mu = self.mu(x)

        return mu


class ActorSigmaNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorSigmaNetwork, self).__init__()

        # Layers
        self.fc1 = nn.Linear(input_shape[-1], n_features)
        self.fc2 = nn.Linear(n_features, n_features)
        self.fc3 = nn.Linear(n_features, output_shape[0])

        # Initialization
        nn.init.xavier_uniform_(
            self.fc1.weight,
            gain=nn.init.calculate_gain('relu')
        )
        nn.init.xavier_uniform_(
            self.fc2.weight,
            gain=nn.init.calculate_gain('relu')
        )
        nn.init.xavier_uniform_(
            self.fc3.weight,
            gain=nn.init.calculate_gain('linear')
        )

    def forward(self, state):
        state = torch.squeeze(state, 1).float()
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # we choose a linear activation function for sigma,
        # since the output is considered as the log of the variance,
        # and therefore it can be any real number
        sigma = self.fc3(x)

        return sigma


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(CriticNetwork, self).__init__()

        # Layers
        self.fc1 = nn.Linear(input_shape[-1], n_features)
        self.fc2 = nn.Linear(n_features, n_features)
        self.fc3 = nn.Linear(n_features, 1)

        # Initialization
        nn.init.xavier_uniform_(
            self.fc1.weight,
            gain=nn.init.calculate_gain('relu')
        )
        nn.init.xavier_uniform_(
            self.fc2.weight,
            gain=nn.init.calculate_gain('relu')
        )
        nn.init.xavier_uniform_(
            self.fc3.weight,
            gain=nn.init.calculate_gain('linear')
        )

    def forward(self, state, action):
        x = torch.cat((state.float(), action.float()), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)

        return torch.squeeze(q)


def create_sac_agent(
        mdp,
        n_features_actor,
        lr_actor,
        n_features_critic,
        lr_critic,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
        log_std_min=-20,
        log_std_max=2,
        target_entropy=None,
        critic_fit_params=None
):

    actor_mu_params = dict(
        network=ActorMuNetwork,
        input_shape=mdp.info.observation_space.shape,
        output_shape=mdp.info.action_space.shape,
        n_features=n_features_actor
    )

    actor_sigma_params = dict(
        network=ActorSigmaNetwork,
        input_shape=mdp.info.observation_space.shape,
        output_shape=mdp.info.action_space.shape,
        n_features=n_features_actor
    )

    optimizer = {
        'class': optim.Adam,
        'params': {'lr': lr_actor}
    }

    critic_params = {
        'network': CriticNetwork,
        'input_shape': (mdp.info.observation_space.shape[0] + mdp.info.action_space.shape[0],),
        'output_shape': (1,),
        'n_features': n_features_critic,
        'optimizer': {
            'class': optim.Adam,
            'params': {'lr': lr_critic}
        },
        'loss': F.mse_loss,
    }

    return SAC(
        mdp.info,
        actor_mu_params,
        actor_sigma_params,
        optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
        log_std_min,
        log_std_max,
        target_entropy,
        critic_fit_params
    )


def run_sac_training(
        core,
        n_epochs,
        n_steps_learn=600,
        n_steps_test=400,
        n_steps_per_fit=1,
        initial_replay_size=500,
        gamma_eval=1,
        record=False,
        record_every=10,
        n_recordings=1,
        record_postfix=''
):
    # Fill the replay memory with random samples
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, quiet=True)
    data = [compute_metrics(core.evaluate(n_steps=n_steps_test, render=False, quiet=True), gamma_eval)]

    progressbar = tqdm(range(n_epochs), unit='epoch')
    for n in progressbar:
        core.learn(n_steps=n_steps_learn, n_steps_per_fit=n_steps_per_fit, quiet=True)
        data.append(compute_metrics(core.evaluate(n_steps=n_steps_test, render=False, quiet=True)))

        if record and (n+1) % record_every == 0:
            for i in range(n_recordings):
                core.evaluate(n_episodes=1, quiet=True)
                core.mdp.render(f"Epo {n+1} - Eval {i+1}{record_postfix}")
            core.agent.save(f"weights/agent_epo_{n+1}{record_postfix}")

        progressbar.set_postfix(MinMaxMean=np.round(data[-1][:3], 2))

    return np.array(data)

