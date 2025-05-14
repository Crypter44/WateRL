import matplotlib as mpl
import numpy as np
from tqdm import tqdm

from Mushroom.agents.agent_factory import create_ddpg_agent
from Mushroom.agents.sigma_decay_policies import set_noise_for_all, update_sigma_for_all
from Mushroom.core.multi_agent_core_labeled import MultiAgentCoreLabeled
from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Mushroom.utils.utils import set_seed, compute_metrics_with_labeled_dataset, final_evaluation
from Mushroom.utils.wandb_handler import wandb_training, create_log_dict

# PARAMS
config = dict(
    seed=0,
    gamma=0.99,

    lr_actor=2e-4,
    lr_critic=1e-3,
    critic_warmup=True,

    initial_replay_size=5000,
    max_replay_size=60000,
    batch_size=200,

    n_features=80,
    tau=.005,

    sigma=[(0, 1), (1, 0.075), (50, 0.025), (70, 0.01)],
    decay_type="exponential",

    n_epochs=100,
    n_episodes_learn=200,
    n_episodes_test=6,
    n_steps_per_fit=1,

    num_agents=2,
    n_episodes_final=1000,
    n_episodes_final_render=200,
    n_epochs_per_checkpoint=20,

    state_selector=[
        0, 1, 2, 3
    ],

    observation_selector=[
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ],

    criteria={
        "demand": {
            "w": 10.0,
            "bound": 0.05,
            "value_at_bound": 0.001,
            "max": 0,
            "min": -1
        },
        "power_per_flow": {"w": .1},
        "negative_flow": {
            "w": 3.0,
            "threshold": 0,
        },
        "target_opening": {
            "max": 1,
            "min": 0,
            "w": 10.0,
            "target": 0.9,
            "left_bound": 0.2,
            "value_at_left_bound": 0.05,
            "right_bound": 0.05,
            "value_at_right_bound": 0.001,
        }
    },
    demand=("uniform_global", 0.4, 1.4)
)
# END_PARAMS

mpl.rcParams['figure.max_open_warning'] = -1


def train(run, save_path, **kwargs):
    set_seed(run.config.seed)
    # MDP
    mdp = CircularFluidNetwork(
        gamma=run.config.gamma,
        criteria=run.config.criteria,
        demand=run.config.demand,
        labeled_step=True,
        state_selector=run.config.state_selector,
        observation_selector=run.config.observation_selector,
    )
    agents = [create_ddpg_agent(
        mdp,
        agent_idx=i,
        n_features_actor=run.config.n_features,
        lr_actor=run.config.lr_actor,
        n_features_critic=run.config.n_features,
        lr_critic=run.config.lr_critic,
        batch_size=run.config.batch_size,
        initial_replay_size=run.config.initial_replay_size,
        max_replay_size=run.config.max_replay_size,
        tau=run.config.tau,
        sigma_checkpoints=run.config.sigma,
        decay_type=run.config.decay_type,
    ) for i in range(run.config.num_agents)]

    # Core
    core = MultiAgentCoreLabeled(agents=agents, mdp=mdp)

    # Fill replay buffer
    core.learn(
        n_steps=run.config.initial_replay_size,
        n_steps_per_fit_per_agent=[run.config.initial_replay_size] * run.config.num_agents,
    )

    # Initial evaluation for comparison
    core.evaluate(n_episodes=1, render=False, quiet=True)
    core.mdp.render(save_path=save_path + f"Epoch_0_Noisy")
    set_noise_for_all(core.agents, False)
    score = compute_metrics_with_labeled_dataset(core.evaluate(n_episodes=run.config.n_episodes_test, render=False)[0])
    set_noise_for_all(core.agents, True)
    core.mdp.render(save_path=save_path + f"Epoch_0")

    if run.config.critic_warmup:
        # First fit
        core.learn(
            n_episodes=run.config.n_episodes_learn,
            n_steps_per_fit_per_agent=[run.config.n_steps_per_fit] * run.config.num_agents,
        )

        # Reset replay memory and noise
        for a in agents:
            a._replay_memory.reset()
        update_sigma_for_all(core.agents, "next")

        # Refill replay memory with random dataset, but with lower noise
        core.learn(
            n_steps=run.config.initial_replay_size,
            n_steps_per_fit_per_agent=[run.config.initial_replay_size] * run.config.num_agents,
        )

    pbar = tqdm(range(run.config.n_epochs), unit='epoch', leave=False)
    for n in pbar:
        if kwargs.get('alive', None) is not None:
            kwargs['alive'].set()

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

        # Update sigma
        update_sigma_for_all(agents)

        # Save checkpoint if necessary
        if (n + 1) % run.config.n_epochs_per_checkpoint == 0:
            for i, a in enumerate(core.agents):
                a.save(save_path + f"/checkpoints/Epoch_{n + 1}_Agent_{i}")
    if kwargs.get('safe', None) is not None:
        kwargs['alive'].set()
        kwargs['safe'].set()
    set_noise_for_all(core.agents, False)
    final = final_evaluation(run.config.n_episodes_final, run.config.n_episodes_final_render, core, save_path)

    # Log final metrics dict for wandb
    run.summary.update({"final": final})

    return


wandb_training(
    project="YourProjectName", # TODO change to your wandb settings
    group="YourGroupName",
    train=train,
    base_path="./Results_Full/", # TODO change to your path
    base_config=config,
    params={
        'seed': [0, 42, 420, 4366, 7359, 738377, 873737],
    },
    notes="""Training with full observability""",
    time_limit_in_sec=60 * 40, # TODO change if needed, on slower machines this might interfere with the training
    inactivity_timeout=180 # TODO see above
)
