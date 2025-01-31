import inspect
import json

from mushroom_rl.algorithms.actor_critic import DDPG
from torch import optim
from torch.nn import functional as F

from Mushroom.agents.mixerddpg import MixerDDPG
from Mushroom.agents.facmac import FACMAC
from Mushroom.agents.maddpg import MADDPG
from Mushroom.agents.maddpg_unified_critic import UnifiedCriticMADDPG
from Mushroom.agents.networks import ActorNetwork, CriticNetwork, MADDPGCriticNetwork
from Mushroom.agents.sigma_decay_policies import UnivariateGaussianPolicy
from Mushroom.utils.replay_memories import ReplayMemoryObsMultiAgent


def create_ddpg_agent(
        mdp,
        agent_idx=-1,
        n_features_actor=80,
        lr_actor=1e-4,
        n_features_critic=80,
        lr_critic=1e-3,
        batch_size=200,
        initial_replay_size=500,
        max_replay_size=5000,
        tau=0.001,
        sigma_checkpoints=None,
        decay_type='exponential',
        sigma=None,
        target_sigma=None,
        sigma_transition_length=None,
        ma_critic=False,
        save_path=None
):
    # Approximator
    actor_input_shape = mdp.local_observation_space(agent_idx).shape
    actor_params = dict(network=ActorNetwork,
                        n_features=n_features_actor,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space_for_idx(agent_idx).shape,
                        agent_idx=agent_idx)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': lr_actor}}

    if ma_critic:
        critic_input_shape = actor_input_shape[0]
        for i in range(mdp.info.n_agents):
            critic_input_shape += mdp.local_action_space(agent_idx).shape[0]
        critic_input_shape = (critic_input_shape,)
    else:
        critic_input_shape = (actor_input_shape[0] + mdp.local_action_space(agent_idx).shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic}},
                         loss=F.mse_loss,
                         n_features=n_features_critic,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         agent_idx=agent_idx,
                         ma_critic=ma_critic)

    policy_class = UnivariateGaussianPolicy
    policy_params = dict(initial_sigma=sigma, target_sigma=target_sigma, decay_type=decay_type,
                         updates_till_target_reached=sigma_transition_length, sigma_checkpoints=sigma_checkpoints)

    if save_path is not None:
        with open(save_path + f"Agent_{agent_idx}.json", "w") as f:
            frame = inspect.currentframe()
            args_as_dict = inspect.getargvalues(frame).locals
            del args_as_dict['frame']  # Avoid including the frame reference
            json.dump({key: str(value) for key, value in args_as_dict.items()}, f, indent=4, )

    if not ma_critic:
        agent = DDPG(mdp.info, policy_class, policy_params,
                          actor_params, actor_optimizer, critic_params,
                          batch_size, initial_replay_size, max_replay_size,
                          tau)
    else:
        agent = MADDPG(mdp.info, agent_idx, policy_class, policy_params,
                                    actor_params, actor_optimizer, critic_params,
                                    batch_size, initial_replay_size, max_replay_size,
                                    tau)
    return agent


def setup_maddpg_agents(
        n_agents,
        mdp,
        n_features_actor=80,
        lr_actor=1e-4,
        n_features_critic=80,
        lr_critic=1e-3,
        batch_size=200,
        initial_replay_size=500,
        max_replay_size=5000,
        tau=0.001,
        sigma_checkpoints=None,
        decay_type='exponential',
        sigma=None,
        target_sigma=None,
        sigma_transition_length=None,
        save_path=None
):
    agents = []
    for i in range(n_agents):
        agents.append(
            create_ddpg_agent(
                mdp, i, n_features_actor, lr_actor, n_features_critic, lr_critic, batch_size,
                initial_replay_size, max_replay_size, tau, sigma_checkpoints, decay_type, sigma, target_sigma,
                sigma_transition_length, True, save_path
            )
        )

    for a in agents:
        a.agents = agents

    return agents


def setup_maddpg_agents_with_unified_critic(
        mdp,
        n_agents,
        policy=None,
        n_features_actor=80,
        lr_actor=1e-4,
        n_features_critic=80,
        lr_critic=2e-4,
        batch_size=200,
        initial_replay_size=500,
        max_replay_size=5000,
        tau=0.001,
        sigma=0.2,
        target_sigma=0.001,
        sigma_transition_length=1,
):
    agents = []
    for i in range(n_agents):
        actor_input_shape = mdp.info.observation_space_for_idx(i).shape
        actor_params = dict(
            network=ActorNetwork,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr_actor}
            },
            n_features=n_features_actor,
            input_shape=actor_input_shape,
            output_shape=mdp.info.action_space_for_idx(i).shape,
            agent_idx=i,
            use_cuda=True,
        )

        critic_input_shape = (actor_input_shape[0] + mdp.info.action_space_for_idx(i).shape[0],)
        critic_params = dict(
            network=CriticNetwork,
            optimizer={'class': optim.Adam,
                       'params': {'lr': lr_critic}},
            loss=F.mse_loss,
            n_features=n_features_critic,
            input_shape=critic_input_shape,
            output_shape=(1,),
            agent_idx=i,
            use_cuda=True
        )

        if policy is None:
            policy = UnivariateGaussianPolicy(
                initial_sigma=sigma,
                target_sigma=target_sigma,
                updates_till_target_reached=sigma_transition_length
            )

        agents.append(
            MixerDDPG(
                mdp.info,
                i,
                policy,
                actor_params,
                critic_params,
                batch_size,
                target_update_frequency=-1,
                tau=tau,
                warmup_replay_size=initial_replay_size,
                replay_memory=None,
                use_cuda=True,
                use_mixer=True,
                primary_agent=None,
            )
        )

    maddpg = UnifiedCriticMADDPG(
        mdp_info=mdp.info,
        batch_size=batch_size,
        replay_memory=ReplayMemoryObsMultiAgent(
            max_size=max_replay_size,
            state_dim=mdp.info.state_space.shape[0],
            obs_dim=[mdp.info.observation_space_for_idx(i).shape[0] for i in range(n_agents)],
            action_dim=[mdp.info.action_space_for_idx(i).shape[0] for i in range(n_agents)],
            n_agents=n_agents,
            discrete_actions=False,
        ),
        target_update_frequency=-1,
        tau=tau,
        warmup_replay_size=initial_replay_size,
        target_update_mode="soft",
        actor_optimizer_params={
            "class": optim.Adam,
            "params": {"lr": lr_actor},
        },
        critic_params={
            "network": MADDPGCriticNetwork,
            "optimizer": {"class": optim.Adam, "params": {"lr": lr_critic}},
            "loss": F.mse_loss,
            "n_features": n_features_critic,
            "input_shape": (mdp.info.state_space.shape[0] + n_agents,),
            "output_shape": (1,),
            "use_cuda": True,
        },
        scale_critic_loss=False,
        scale_actor_loss=False,
        obs_last_action=False,
        host_agents=agents,
        use_cuda=True,
    )

    return agents, maddpg


def setup_facmac_agents(
        mdp,
        n_agents,
        policy=None,
        n_features_actor=80,
        lr_actor=1e-4,
        n_features_critic=80,
        lr_critic=2e-4,
        batch_size=200,
        initial_replay_size=500,
        max_replay_size=5000,
        tau=0.001,
        sigma=0.2,
        target_sigma=0.001,
        sigma_transition_length=1,
):
    agents = []
    for i in range(n_agents):
        actor_input_shape = mdp.info.observation_space_for_idx(i).shape
        actor_params = dict(
            network=ActorNetwork,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr_actor}
            },
            n_features=n_features_actor,
            input_shape=actor_input_shape,
            output_shape=mdp.info.action_space_for_idx(i).shape,
            agent_idx=i,
            use_cuda=True,
        )

        critic_input_shape = (actor_input_shape[0] + mdp.info.action_space_for_idx(i).shape[0],)
        critic_params = dict(
            network=CriticNetwork,
            optimizer={'class': optim.Adam,
                       'params': {'lr': lr_critic}},
            loss=F.mse_loss,
            n_features=n_features_critic,
            input_shape=critic_input_shape,
            output_shape=(1,),
            agent_idx=i,
            use_cuda=True
        )

        if policy is None:
            policy = UnivariateGaussianPolicy(
                initial_sigma=sigma,
                target_sigma=target_sigma,
                updates_till_target_reached=sigma_transition_length
            )

        agents.append(
            MixerDDPG(
                mdp.info,
                i,
                policy,
                actor_params,
                critic_params,
                batch_size,
                target_update_frequency=-1,
                tau=tau,
                warmup_replay_size=initial_replay_size,
                replay_memory=None,
                use_cuda=True,
                use_mixer=True,
                primary_agent=None,
            )
        )

    facmac = FACMAC(
        mdp.info,
        batch_size,
        replay_memory=ReplayMemoryObsMultiAgent(
            max_size=max_replay_size,
            state_dim=mdp.info.state_space.shape[0],
            obs_dim=[mdp.info.observation_space_for_idx(i).shape[0] for i in range(n_agents)],
            action_dim=[mdp.info.action_space_for_idx(i).shape[0] for i in range(n_agents)],
            n_agents=n_agents,
            discrete_actions=False,
        ),
        target_update_frequency=-1,
        tau=tau,
        warmup_replay_size=initial_replay_size,
        target_update_mode="soft",
        mixing_embed_dim=32,
        actor_optimizer_params={
            'class': optim.Adam,
            'params': {'lr': lr_actor}
        },
        critic_optimizer_params={
            'class': optim.Adam,
            'params': {'lr': lr_critic}
        },
        scale_critic_loss=False,
        scale_actor_loss=False,
        grad_norm_clip=0.5,
        obs_last_action=False,
        host_agents=agents,
        use_cuda=True
    )

    return agents, facmac
