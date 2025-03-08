from tqdm import tqdm

import Mushroom.environments.test.constant_value_env_aryaman as env
import aryaman
from Mushroom.agents.agent_factory import create_ddpg_agent
from Mushroom.agents.sigma_decay_policies import set_noise_for_all, update_sigma_for_all
from Mushroom.utils.utils import compute_metrics_with_labeled_dataset, set_seed
from Mushroom.utils.wandb_handler import wandb_training

# PARAMS
config = dict(
    n_agents=1,
    mdp_params=dict(
        gamma=0.99,
        target=0.4,
        bound=0.5,
        value_at_bound=0.001,
        reward_fn='exponential',
        min=0,
        max=1,
        absorbing=False,
    ),
    actor_n_features=80,
    actor_lr=1e-3,
    actor_activation='sigmoid',
    critic_n_features=80,
    critic_lr=1e-2,
    critic_tau=0.25,
    batch_size=200,
    n_steps_per_fit=1,
    initial_replay_size=5_000,
    max_replay_size=25_000,
    tau=0.001,
    sigma_checkpoints=[(0, 0.5), (200, 0.1), (400, 0.05), (600, 0.01)],
    n_epochs=1500,
    seed=0,
)


# END_PARAMS


def train(run, save_path):
    set_seed(run.config.seed)
    mdp = env.Plug(n_agents=run.config.n_agents, **run.config.mdp_params)

    agents = [create_ddpg_agent(
        mdp,
        i,
        run.config.actor_n_features,
        run.config.actor_lr,
        run.config.actor_activation,
        run.config.critic_n_features,
        run.config.critic_lr,
        run.config.batch_size,
        run.config.initial_replay_size,
        run.config.max_replay_size,
        run.config.tau,
        run.config.sigma_checkpoints
    ) for i in range(run.config.n_agents)]
    for agent in agents:
        agent.set_debug_logging(True)
        agent._critic_tau = run.config.critic_tau

    core = aryaman.MultiAgentCore(agents, mdp)
    core.learn(
        n_steps=run.config.initial_replay_size,
        n_steps_per_fit_per_agent=[run.config.initial_replay_size] * run.config.n_agents
    )

    pbar = tqdm(range(run.config.n_epochs), leave=False)
    for _ in pbar:

        # Train
        mdp.reset_actions()
        core.learn(n_steps=500, n_steps_per_fit_per_agent=[run.config.n_steps_per_fit] * run.config.n_agents)
        train_actions = mdp.get_debug_info()

        # Eval
        mdp.reset_actions()
        set_noise_for_all(agents, False)
        eval_data, _ = core.evaluate(n_steps=100)
        score = compute_metrics_with_labeled_dataset(eval_data)
        set_noise_for_all(agents, True)

        # Log
        log = {
            "score/score": score[2],
            "score/min_score": score[0],
            "score/max_score": score[1],
        }
        for i in range(run.config.n_agents):
            log |= {
                f"agent_{i}/{key}": value for key, value in agents[i].get_debug_info(entries_as_list=False).items()
            }
            log |= {
                f"agent_{i}/sigma": agents[i].policy.get_sigma()
            }
        log |= {
            f"mdp/eval/{key}": value for key, value in mdp.get_debug_info().items()
        }
        log |= {
            f"mdp/train/{key}": value for key, value in train_actions.items()
        }
        run.log(log)

        # Update noise
        update_sigma_for_all(agents)
    pbar.close()

    return


wandb_training(
    project="debug_grad_norm",
    group="LowTauForCritic",
    base_config=config,
    base_path="/home/js/Downloads/Aryaman",
    train=train,
    params={
        "mdp_params.target": [0.25, 0.75],
    },
    notes="""
    Testing the new wandb training function with an actual use case.
    We have different tau values for actor and critic. This enables us to speed up the critic learning process. 
    """,
)
