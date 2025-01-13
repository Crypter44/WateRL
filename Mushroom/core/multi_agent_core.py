from tqdm import tqdm

import numpy as np


class Core(object):
    """
    Implements the functions to run a generic algorithm.

    """

    def __init__(
        self, agent, mdp, callbacks_fit=None, callback_step=None, preprocessors=None
    ):
        """
        Constructor.

        Args:
            agent (Agent): (list of) agents moving according to a policy;
            mdp (Environment): the environment in which the agent moves;
            callbacks_fit (list): list of callbacks to execute at the end of
                each fit;
            callback_step (Callback): callback to execute after each step;
            preprocessors (list): list of state preprocessors to be
                applied to state variables before feeding them to the
                agent.

        """
        self.agent = agent
        self.mdp = mdp
        self.callbacks_fit = callbacks_fit if callbacks_fit is not None else list()
        self.callback_step = (
            callback_step if callback_step is not None else lambda x: None
        )
        self._preprocessors = preprocessors if preprocessors is not None else list()

        self._state = None

        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter_per_agent = np.zeros(len(self.agent), dtype=int)
        self._current_steps_counter_per_agent = np.zeros(len(self.agent), dtype=int)
        self._episode_steps = None
        self._n_episodes = None
        self._n_steps_per_fit_per_agent = None
        self._n_episodes_per_fit_per_agent = None

        self.action_norms = [[] for _ in range(len(agent))]

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
                    lambda idx_agent=idx_agent_loop, n_steps_per_fit=n_steps_per_fit_loop: self._current_steps_counter_per_agent[
                        idx_agent
                    ]
                    >= n_steps_per_fit
                )
        else:  # train every no. episodes
            for idx_agent_loop, n_episodes_per_fit_loop in enumerate(
                self._n_episodes_per_fit_per_agent
            ):
                fit_condition_per_agent.append(
                    lambda idx_agent=idx_agent_loop, n_episodes_per_fit=n_episodes_per_fit_loop: self._current_episodes_counter_per_agent[
                        idx_agent
                    ]
                    >= n_episodes_per_fit
                )
        self._run(n_steps, n_episodes, fit_condition_per_agent, render, quiet)

    def evaluate(
        self,
        initial_states=None,
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
            initial_states (np.ndarray, None): the starting states of each
                episode;
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        fit_condition_per_agent = list()
        for idx_agent in range(len(self.agent)):
            fit_condition_per_agent.append(lambda: False)

        return self._run(
            n_steps, n_episodes, fit_condition_per_agent, render, quiet, initial_states
        )

    def _run(
        self,
        n_steps,
        n_episodes,
        fit_condition_per_agent,
        render,
        quiet,
        initial_states=None,
    ):
        assert (
            n_episodes is not None
            and n_steps is None
            and initial_states is None
            or n_episodes is None
            and n_steps is not None
            and initial_states is None
            or n_episodes is None
            and n_steps is None
            and initial_states is not None
        )

        self._n_episodes = (
            len(initial_states) if initial_states is not None else n_episodes
        )

        if n_steps is not None:
            move_condition = lambda: self._total_steps_counter < n_steps

            steps_progress_bar = tqdm(
                total=n_steps, dynamic_ncols=True, disable=quiet, leave=False
            )
            episodes_progress_bar = tqdm(disable=True)
        else:
            move_condition = lambda: self._total_episodes_counter < self._n_episodes

            steps_progress_bar = tqdm(disable=True)
            episodes_progress_bar = tqdm(
                total=self._n_episodes, dynamic_ncols=True, disable=quiet, leave=False
            )

        return self._run_impl(
            move_condition,
            fit_condition_per_agent,
            steps_progress_bar,
            episodes_progress_bar,
            render,
            initial_states,
        )

    def _run_impl(
        self,
        move_condition,
        fit_condition,
        steps_progress_bar,
        episodes_progress_bar,
        render,
        initial_states,
    ):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0

        dataset = list()
        last = True
        while move_condition():
            if last:
                self.reset(initial_states)

            sample = self._step(render)

            self.callback_step([sample])

            self._total_steps_counter += 1
            self._current_steps_counter += 1
            steps_progress_bar.update(1)

            if sample[-1]:
                self._total_episodes_counter += 1
                self._current_episodes_counter += 1
                episodes_progress_bar.update(1)

            dataset.append(sample)
            if fit_condition():
                self.agent.fit(dataset)
                self._current_episodes_counter = 0
                self._current_steps_counter = 0

                for c in self.callbacks_fit:
                    c(dataset)

                dataset = list()

            last = sample[-1]

        self.agent.stop()
        self.mdp.stop()

        steps_progress_bar.close()
        episodes_progress_bar.close()

        return dataset

    def _step(self, render):
        """
        Single step.

        Args:
            render (bool): whether to render or not.

        Returns:
            A tuple containing the previous state, the action sampled by the
            agent, the reward obtained, the reached state, the absorbing flag
            of the reached state and the last step flag.

        """
        action = [self.agent[0].draw_action(self._state)]
        next_state, reward, absorbing, _ = self.mdp.step(action)

        self._episode_steps += 1

        if render:
            render_info = {"action": action, "reward": reward}
            self.mdp.render(render_info)

        last = not (self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state

        return state, action, reward, next_state, absorbing, last

    def reset(self, initial_states=None):
        """
        Reset the state of the agent.

        """
        if initial_states is None or self._total_episodes_counter == self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        self._state = self._preprocess(self.mdp.reset(initial_state).copy())
        self.agent.episode_start()
        self.agent.next_action = None
        self._episode_steps = 0

    def _preprocess(self, state):
        """
        Method to apply state preprocessors.

        Args:
            state (np.ndarray): the state to be preprocessed.

        Returns:
             The preprocessed state.

        """
        for p in self._preprocessors:
            state = p(state)

        return state


class MultiAgentCore(Core):
    def _run_impl(
        self,
        move_condition,
        fit_condition_per_agent,
        steps_progress_bar,
        episodes_progress_bar,
        render,
        initial_states,
    ):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter_per_agent = np.zeros(len(self.agent), dtype=int)
        self._current_steps_counter_per_agent = np.zeros(len(self.agent), dtype=int)

        dataset_per_agent = [list() for _ in range(len(self.agent))]
        last = True
        while move_condition():
            if last:
                self.reset(initial_states)

            sample = self._step(render)

            self.callback_step([sample])

            self._total_steps_counter += 1
            self._current_steps_counter_per_agent += 1
            steps_progress_bar.update(1)

            last = sample[-1]
            if last:
                self._total_episodes_counter += 1
                self._current_episodes_counter_per_agent += 1
                episodes_progress_bar.update(1)

            [
                dataset.append(sample)
                for idx_agent, dataset in enumerate(dataset_per_agent)
            ]

            for idx_agent, fit_condition in enumerate(fit_condition_per_agent):
                if fit_condition():
                    self.agent[idx_agent].fit(dataset_per_agent[idx_agent])
                    self._current_episodes_counter_per_agent[idx_agent] = 0
                    self._current_steps_counter_per_agent[idx_agent] = 0

                    if idx_agent == 0:
                        for c in self.callbacks_fit:
                            c(dataset_per_agent[0])
                    else:
                        pass
                    dataset_per_agent[
                        idx_agent
                    ] = (
                        list()
                    )  # fit stores data in agent replay buffer, so core's replay buffer can be reset

        for agent in self.agent:
            agent.stop()
        self.mdp.stop()

        steps_progress_bar.close()
        episodes_progress_bar.close()

        return dataset_per_agent[0]  # just protagonist dataset

    def _step(self, render):
        action = list()
        for idx_agent in range(len(self.agent)):
            action.append(self.agent[idx_agent].draw_action(self._state))
            self.action_norms[idx_agent].append(np.linalg.norm(action[idx_agent]))

        next_state, reward, absorbing, info = self.mdp.step(action)

        self._episode_steps += 1

        if render:
            render_info = {"action": action, "reward": reward}
            self.mdp.render(render_info)

        last = not (self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state

        return state, action, reward, next_state, absorbing, last

    def reset(self, initial_states=None):
        """
        Reset the state of the mdp and agents.

        """
        if initial_states is None or self._total_episodes_counter == self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        self._state = self._preprocess(self.mdp.reset(initial_state).copy())
        for agent in self.agent:
            agent.episode_start()
            agent.next_action = None
        self._episode_steps = 0
