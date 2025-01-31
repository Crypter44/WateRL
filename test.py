import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from Mushroom.agents.mixerddpg import MixerDDPG
from aryaman import Environment, MDPInfo, Box, GaussianPolicy, ContinuousActorNetwork, ContinuousCriticNetwork, \
    ReplayMemoryObs, MultiAgentCore, compute_J_all_agents


def set_seed(seed: int):
    """
    Set the seed of the random number generators of numpy, torch and random.
    :param seed: The seed to set.
    :return: None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


set_seed(1)


class CustomEnv(Environment):
    def __init__(self):
        super().__init__(
            MDPInfo(
                state_space=Box(low=-1, high=1, shape=(2,)),
                observation_space=Box(low=-1, high=1, shape=(2,)),
                action_space=Box(low=-1, high=1, shape=(1,)),
                discrete_actions=False,
                gamma=0.99,
                horizon=100,
                dt=0.1,
                has_obs=True,
                has_action_masks=False,
                n_agents=2,
            )
        )

    def reset(self):
        return {
            "state": np.array([0, 0]),
            "obs": [np.array([0, 0])] * 2,
            "absorbing": False,
            "info": {}
        }

    def step(self, action):
        reward = np.exp(-5 * (np.mean(action) - 0.6) ** 2)
        return {
            "state": np.array([0, 0]),
            "obs": [np.array([0, 0])] * 2,
            "rewards": [reward] * 2,
            "absorbing": False,
            "info": {}
        }


mdp = CustomEnv()
agents = [MixerDDPG(
    mdp_info=mdp.info,
    idx_agent=i,
    policy=GaussianPolicy(np.array([0.4]), Box(low=-1, high=1, shape=(1,))),
    actor_params=dict(
        network=ContinuousActorNetwork,
        optimizer={
            'class': optim.Adam,
            'params': {'lr': 1e-5}
        },
        input_shape=(2,),
        output_shape=(1,),
        n_features=(80, 80),
        use_cuda=False,
    ),
    critic_params=dict(
        network=ContinuousCriticNetwork,
        optimizer={
            'class': optim.Adam,
            'params': {'lr': 2e-5}
        },
        loss=F.mse_loss,
        input_shape=(3,),
        output_shape=(1,),
        n_features=(80, 80),
        use_cuda=False,
    ),
    batch_size=200,
    target_update_frequency=-1,
    tau=0.001,
    warmup_replay_size=500,
    replay_memory=ReplayMemoryObs(
        5000,
        2,
        2,
        1,
        False,
    ),
    use_cuda=False,
    primary_agent=None,
    use_mixer=False,
) for i in range(2)]

core = MultiAgentCore(agents=agents, mdp=mdp)

for a in agents:
    a.policy.set_mode("test")
d, _ = core.evaluate(n_steps=300)
for a in agents:
    a.policy.set_mode("train")
J = compute_J_all_agents(d)
print("Min J: ", np.min(J))
print("Max J: ", np.max(J))
print("Mean J: ", np.mean(J))

core.learn(n_steps=500, n_steps_per_fit_per_agent=[500] * 2, quiet=False)

for i in range(3):
    core.learn(n_steps=1500, n_steps_per_fit_per_agent=[1] * 2, quiet=False)
    for a in agents:
        a.policy.set_mode("test")
    d, _ = core.evaluate(n_steps=300)
    for a in agents:
        a.policy.set_mode("train")
    J = compute_J_all_agents(d)
    print("Min J: ", np.min(J))
    print("Max J: ", np.max(J))
    print("Mean J: ", np.mean(J))

