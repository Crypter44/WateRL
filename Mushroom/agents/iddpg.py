from copy import deepcopy

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import to_parameter

from Mushroom.environments.mdp_info import MAMDPInfo
from Mushroom.utils.replay_memories import ReplayMemoryObsMultiAgent


class IDDPG(DeepAC):
    def __init__(
            self,
            agent_idx: int,
            mdp_info: MAMDPInfo,
            policy_class, policy_params,
            actor_params, actor_optimizer,
            critic_params,
            batch_size,
            initial_replay_size, max_replay_size,
            tau,
            policy_delay=1,
            critic_fit_params=None,
            actor_predict_params=None,
            critic_predict_params=None
    ):
        """
        """
        self.agent_idx = agent_idx
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params
        self._actor_predict_params = dict() if actor_predict_params is None else actor_predict_params
        self._critic_predict_params = dict() if critic_predict_params is None else critic_predict_params

        self._batch_size = to_parameter(batch_size)
        self._tau = to_parameter(tau)
        self._policy_delay = to_parameter(policy_delay)
        self._fit_count = 0

        self._initial_replay_size = initial_replay_size
        self._replay_memory = ReplayMemoryObsMultiAgent(
            max_replay_size,
            mdp_info.state_space.shape[0],
            [mdp_info.observation_space_for_idx(i).shape[0] for i in range(mdp_info.n_agents)],
            [mdp_info.action_space_for_idx(i).shape[0] for i in range(mdp_info.n_agents)],
            mdp_info.n_agents,
            False
        )

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        target_actor_params = deepcopy(actor_params)
        self._actor_approximator = Regressor(TorchApproximator,
                                             **actor_params)
        self._target_actor_approximator = Regressor(TorchApproximator,
                                                    **target_actor_params)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)
        self._init_target(self._actor_approximator,
                          self._target_actor_approximator)

        policy = policy_class(self._actor_approximator, **policy_params)

        policy_parameters = self._actor_approximator.model.network.parameters()

        self._add_save_attr(
            _critic_fit_params='pickle',
            _critic_predict_params='pickle',
            _actor_predict_params='pickle',
            _batch_size='mushroom',
            _tau='mushroom',
            _policy_delay='mushroom',
            _fit_count='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _target_actor_approximator='mushroom'
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset, **info):
        self._replay_memory.add(dataset)
        if self._replay_memory.size > self._initial_replay_size:
            states, obs, actions, rewards, next_states, next_obs, absorbing, last = \
                self._replay_memory.get(self._batch_size())

            q_next = self._next_q(next_states, next_obs, absorbing)
            q = rewards[:, self.agent_idx] + self.mdp_info.gamma * q_next

            self._fit_critic(states, obs, actions, q)

            if self._fit_count % self._policy_delay() == 0:
                loss = self._loss(states, obs)
                self._optimize_actor_parameters(loss)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)
            self._update_target(self._actor_approximator,
                                self._target_actor_approximator)

            self._fit_count += 1

        return -1, -1

    def _loss(self, states, obs):
        obs = obs[self.agent_idx]
        action = self._actor_approximator(obs, output_tensor=True, **self._actor_predict_params)
        q = self._critic_approximator(obs, action, output_tensor=True, **self._critic_predict_params)

        return -q.mean()

    def _next_q(self, next_states, next_obs, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        next_obs = next_obs[self.agent_idx]
        a = self._target_actor_approximator.predict(next_obs, **self._actor_predict_params)

        q = self._target_critic_approximator.predict(next_obs, a, **self._critic_predict_params)
        q *= 1 - absorbing

        return q

    def _fit_critic(self, states, obs, actions, q):
        self._critic_approximator.fit(obs[self.agent_idx], actions[self.agent_idx], q,
                                      **self._critic_fit_params)

    def _post_load(self):
        self._actor_approximator = self.policy._approximator
        self._update_optimizer_parameters(self._actor_approximator.model.network.parameters())
