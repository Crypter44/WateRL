import numpy as np
from tqdm import tqdm

from Mushroom.agents.qmix import Mixer
from Mushroom.environments.environment import MAEnvironment


class MultiAgentCoreMixer(object):
    def __init__(
            self,
            agents: list = None,
            mixer: Mixer = None,
            mdp: MAEnvironment = None,
            callbacks_step: dict = {},
    ):
        """
        Constructor.

        Args:
            agents (Agent): list of agents moving according to a policy;
            mdp (Environment): the environment in which the agent moves;
            callbacks_step (dict): dict of callbacks to execute after each step
        """
        self.agents = agents
        self.mixer = mixer
        self.mdp = mdp
        self.callbacks_step = callbacks_step
        self.obs_last_action = getattr(agents[0], "_obs_last_action", False)

        self._state = None
        self._obs = None
        self._action_masks = None

        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter_per_agent = np.zeros(
            self.mdp.info.n_agents, dtype=int
        )
        self._current_steps_counter_per_agent = np.zeros(
            self.mdp.info.n_agents, dtype=int
        )
        self._episode_steps = None

    def learn(
            self,
            n_steps=None,
            n_episodes=None,
            n_steps_per_fit_per_agent=None,
            n_episodes_per_fit_per_agent=None,
            render=False,
            quiet=False,
    ):
        """
        This function moves the agent in the environment and fits the policy
        using the collected samples. The agent can be moved for a given number
        of steps or a given number of episodes and, independently from this
        choice, the policy can be fitted after a given number of steps or a
        given number of episodes. By default, the environment is reset.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            n_steps_per_fit_per_agent (list, None): number of steps between each fit of each agent's
                policies;
            n_episodes_per_fit_per_agent (list, None): number of episodes between each fit
                of each agent's policies;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        assert (
                       n_episodes_per_fit_per_agent is not None
                       and n_steps_per_fit_per_agent is None
               ) or (
                       n_episodes_per_fit_per_agent is None
                       and n_steps_per_fit_per_agent is not None
               )

        self._n_steps_per_fit_per_agent = n_steps_per_fit_per_agent
        self._n_episodes_per_fit_per_agent = n_episodes_per_fit_per_agent

        fit_condition_per_agent = list()
        if n_steps_per_fit_per_agent is not None:  # train every no. steps
            for idx_agent_loop, n_steps_per_fit_loop in enumerate(n_steps_per_fit_per_agent):
                fit_condition_per_agent.append(
                    lambda idx_agent=idx_agent_loop, n_steps_per_fit=n_steps_per_fit_loop:
                    self._current_steps_counter_per_agent[
                        idx_agent
                    ]
                    >= n_steps_per_fit
                )
        else:  # train every no. episodes
            for idx_agent_loop, n_episodes_per_fit_loop in enumerate(
                    self._n_episodes_per_fit_per_agent
            ):
                fit_condition_per_agent.append(
                    lambda idx_agent=idx_agent_loop, n_episodes_per_fit=n_episodes_per_fit_loop:
                    self._current_episodes_counter_per_agent[
                        idx_agent
                    ]
                    >= n_episodes_per_fit
                )

        need_complete_episodes = (
            True if n_episodes_per_fit_per_agent is not None else False
        )

        self._run(
            n_steps,
            n_episodes,
            fit_condition_per_agent,
            render,
            quiet,
            need_complete_episodes=need_complete_episodes,
        )

    def evaluate(
            self,
            n_steps=None,
            n_episodes=None,
            render=False,
            quiet=False,
    ):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from
        a set of initial states for the whole episode. By default, the
        environment is reset.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """

        fit_condition_per_agent = []
        for _ in range(self.mdp.info.n_agents):
            fit_condition_per_agent.append(lambda: False)

        return self._run(
            n_steps,
            n_episodes,
            fit_condition_per_agent,
            render,
            quiet,
            need_complete_episodes=False,
        )

    def _run(
            self,
            n_steps,
            n_episodes,
            fit_condition_per_agent,
            render,
            quiet,
            need_complete_episodes=False,
    ):
        assert (
                n_episodes is not None
                and n_steps is None
                or n_episodes is None
                and n_steps is not None
        )

        if n_steps is not None:
            move_condition = lambda: self._total_steps_counter < n_steps

            steps_progress_bar = tqdm(
                total=n_steps, dynamic_ncols=True, disable=quiet, leave=False
            )
            episodes_progress_bar = tqdm(disable=True)
        else:
            move_condition = lambda: self._total_episodes_counter < n_episodes

            steps_progress_bar = tqdm(disable=True)
            episodes_progress_bar = tqdm(
                total=n_episodes, dynamic_ncols=True, disable=quiet, leave=False
            )

        dataset, dataset_info = self._run_impl(
            move_condition,
            fit_condition_per_agent,
            steps_progress_bar,
            episodes_progress_bar,
            render,
            need_complete_episodes,
        )
        return dataset, dataset_info

    def _run_impl(
            self,
            move_condition,
            fit_condition_per_agent,
            steps_progress_bar,
            episodes_progress_bar,
            render,
            need_complete_episodes,
    ):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter_per_agent = np.zeros(
            self.mdp.info.n_agents, dtype=int
        )
        self._current_steps_counter_per_agent = np.zeros(
            self.mdp.info.n_agents, dtype=int
        )

        dataset = []
        dataset_info = []
        last = True

        if need_complete_episodes:
            step_count_move_condition = move_condition
            move_condition = lambda: step_count_move_condition() or not last

        while move_condition():
            if last:
                self.reset()

            sample, info = self._step(render)

            self._total_steps_counter += 1
            self._current_steps_counter_per_agent += 1
            steps_progress_bar.update(1)

            last = sample["last"]
            if last:
                self._total_episodes_counter += 1
                self._current_episodes_counter_per_agent += 1
                episodes_progress_bar.update(1)

            dataset.append(sample)

            if all(fit_condition() for fit_condition in fit_condition_per_agent):
                actor_losses = []
                critic_losses = []
                for idx_agent in range(len(self.agents)):
                    actor_loss, critic_loss = self.agents[idx_agent].fit(dataset)
                    self._current_episodes_counter_per_agent[idx_agent] = 0
                    self._current_steps_counter_per_agent[idx_agent] = 0

                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

                self.mixer.fit(dataset)
                dataset = list()
                dataset_info = list()

            self._get_callbacks(sample, info)

        for agent in self.agents:
            agent.stop()
        self.mdp.stop()

        return dataset, dataset_info

    def _step(self, render):
        actions = self._get_actions()

        step = self.mdp.step(actions)

        next_state = step["state"]
        next_obs = step.get("obs", None)
        rewards = step["rewards"]
        absorbing = step["absorbing"]
        next_action_masks = step.get("action_masks", None)
        info = step.get("info", None)

        self._episode_steps += 1

        if self.obs_last_action:
            for i, step_obs in enumerate(next_obs):
                if self.mdp.info.discrete_actions:  # make integer action one-hot
                    action = np.zeros(self.mdp.info.action_space_for_idx[i].n)
                    action[actions[i]] = 1
                else:
                    action = actions[i]
                next_obs[i] = np.concatenate([step_obs, action])

        if render:
            frame = self.mdp.render()

        last = (self._episode_steps >= self.mdp.info.horizon) or absorbing

        obs = self._obs
        next_obs = next_obs.copy()
        self._obs = next_obs

        state = self._state
        next_state = next_state.copy()
        self._state = next_state

        action_masks = self._action_masks.copy() if self._action_masks is not None else None
        self._action_masks = next_action_masks

        sample = {
            "state": state,
            "obs": obs,
            "action_masks": action_masks,
            "actions": actions,
            "rewards": rewards,
            "next_state": next_state,
            "next_obs": next_obs,
            "next_action_masks": next_action_masks,
            "absorbing": absorbing,
            "last": last,
        }

        return sample, info

    def _get_actions(self):
        actions = list()
        for idx_agent in range(self.mdp.info.n_agents):
            if self.mdp.info.has_obs:
                if self.mdp.info.has_action_masks:
                    actions.append(
                        self.agents[idx_agent].draw_action(
                            self._obs[idx_agent], self._action_masks[idx_agent]
                        )
                    )
                else:
                    actions.append(
                        self.agents[idx_agent].draw_action(self._obs[idx_agent])
                    )
            else:
                actions.append(self.agents[idx_agent].draw_action(self._state))
        return actions

    def _get_callbacks(self, sample, info):
        for _, callback in self.callbacks_step.items():
            callback_info = {
                "sample": sample,
                "info": info,
            }
            callback(callback_info)

    def reset(self):
        """
        Reset the state of the mdp and agents.

        """
        init_step = self.mdp.reset()

        self._state = init_step["state"]
        self._obs = init_step.get("obs", None)
        self._action_masks = init_step.get("action_masks", None)

        if self.obs_last_action:
            for i, init_obs in enumerate(self._obs):
                if self.mdp.info.discrete_actions:  # make integer action one-hot
                    action = np.zeros(self.mdp.info.action_space_for_idx[i].n)
                else:
                    action = np.zeros(self.mdp.info.action_space_for_idx[i].shape)
                self._obs[i] = np.concatenate([init_obs, action])

        for agent in self.agents:
            agent.episode_start()
            agent.next_action = None
        self._episode_steps = 0

    def set_random_mode(self):
        for agent in self.agents:
            agent.set_random_mode()

    def set_training_mode(self):
        for agent in self.agents:
            agent.set_training_mode()

    def set_testing_mode(self):
        for agent in self.agents:
            agent.set_testing_mode()

    def set_callbacks_step(self, name, callback):
        self.callbacks_step[name] = callback

    def remove_callbacks_step(self, name):
        self.callbacks_step.pop(name)

    def clear_callbacks_step(self, name):
        self.callbacks_step[name].clear()

    def clear_all_callbacks_step(self):
        for callback in self.callbacks_step.values():
            callback.clear()

    def set_memory_profiler(self, memory_profiler):
        self.memory_profiler = memory_profiler
