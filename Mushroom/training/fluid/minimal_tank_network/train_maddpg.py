import matplotlib as mpl
import numpy as np
from tqdm import tqdm

from Mushroom.agents.agent_factory import setup_maddpg_agents
from Mushroom.agents.sigma_decay_policies import set_noise_for_all, update_sigma_for_all
from Mushroom.core.multi_agent_core_labeled import MultiAgentCoreLabeled
from Mushroom.environments.fluid.minimal_tank_network import MinimalTankNetwork
from Mushroom.utils.utils import set_seed, compute_metrics_with_labeled_dataset, final_evaluation
from Mushroom.utils.wandb_handler import wandb_training, create_log_dict

# PARAMS
config = dict(
    seed=0,
    gamma=0.99,

    lr_actor=1e-3,
    lr_critic=1e-3,

    initial_replay_size=5000,
    max_replay_size=10000,
    batch_size=200,

    n_features=80,
    tau=.005,

    sigma=[(0, 0.8), (10, 0.1)],
    decay_type="exponential",

    n_epochs=30,
    n_episodes_learn=6,
    n_episodes_test=3,
    n_steps_per_fit=1,

    num_agents=2,
    n_episodes_final=1,
    n_episodes_final_render=1,
    n_epochs_per_checkpoint=50,

    state_selector=[
        0, 6, 9
    ],

    observation_selector=[
        [0],
        [9]
    ],

    criteria={
        "demand": {
            "w": 1.0,
            "bound": 0.1,
            "value_at_bound": 0.001,
            "max": 0,
            "min": -1
        },
    },
    demand="tagesgang"
)
# END_PARAMS

mpl.rcParams['figure.max_open_warning'] = -1


def train(run, save_path):
    set_seed(run.config.seed)
    # MDP
    mdp = MinimalTankNetwork(
        gamma=run.config.gamma,
        criteria=run.config.criteria,
        demand_curve=run.config.demand,
        labeled_step=True,
        observation_selector=run.config.observation_selector,
        state_selector=run.config.state_selector,
    )
    agents = setup_maddpg_agents(
        n_agents=run.config.num_agents,
        mdp=mdp,
        n_features_actor=run.config.n_features,
        lr_actor=run.config.lr_actor * run.config.lr_multiplier,
        n_features_critic=run.config.n_features,
        lr_critic=run.config.lr_critic * run.config.lr_multiplier * run.config.critic_lr_multiplier,
        batch_size=run.config.batch_size,
        initial_replay_size=run.config.initial_replay_size,
        max_replay_size=run.config.max_replay_size,
        tau=run.config.tau,
        sigma_checkpoints=run.config.sigma,
        decay_type=run.config.decay_type,
    )

    # Core
    core = MultiAgentCoreLabeled(agents=agents, mdp=mdp)

    # Fill replay buffer
    core.learn(
        n_steps=run.config.initial_replay_size,
        n_steps_per_fit_per_agent=[run.config.initial_replay_size] * run.config.num_agents,
    )

    # Initial evaluation for comparison
    set_noise_for_all(core.agents, False)
    score = compute_metrics_with_labeled_dataset(core.evaluate(n_episodes=run.config.n_episodes_test, render=False)[0])
    set_noise_for_all(core.agents, True)
    core.mdp.render(save_path=save_path + f"Epoch_0")

    pbar = tqdm(range(run.config.n_epochs), unit='epoch', leave=False)
    for n in pbar:
        # Train
        core.learn(
            n_episodes=run.config.n_episodes_learn,
            n_steps_per_fit_per_agent=[run.config.n_steps_per_fit] * run.config.num_agents,
        )

        # Eval
        core.evaluate(n_episodes=1, render=False, quiet=True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}_Noisy")
        set_noise_for_all(core.agents, False)
        dataset, _ = core.evaluate(n_episodes=run.config.n_episodes_test, render=False, )
        set_noise_for_all(core.agents, True)
        core.mdp.render(save_path=save_path + f"Epoch_{n + 1}")
        score = compute_metrics_with_labeled_dataset(dataset)

        # Log
        run.log(create_log_dict(agents, mdp, score))
        pbar.set_postfix(
            MinMaxMean=np.round(score[0:3], 2),
            sigma=np.round(core.agents[0].policy.get_sigma(), 2)
        )

        update_sigma_for_all(agents)

        if (n + 1) % run.config.n_epochs_per_checkpoint == 0:
            for i, a in enumerate(core.agents):
                a.save(save_path + f"/checkpoints/Epoch_{n + 1}_Agent_{i}")

    set_noise_for_all(core.agents, False)
    run.summary.update(
        {
            "final": final_evaluation(
                run.config.n_episodes_final,
                run.config.n_episodes_final_render,
                core,
                save_path
            )
        }
    )

    return


wandb_training(
    project="TankNetworkMADDPG",
    group="InitialRun",
    train=train,
    base_path="./Plots/MADDPG/",
    base_config=config,
    params={
        'seed': [0],
        'lr_multiplier': [0.5],
        'critic_lr_multiplier': [5, 10, 50],
    },
    notes="""Tuning of the DDPG algorithm for the minimal tank network."""
)
