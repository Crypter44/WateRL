from mushroom_rl.core import Serializable


class MAMDPInfo(Serializable):
    """
    This class is used to store the information of the environment.

    """

    def __init__(
        self,
        state_space,
        observation_spaces,
        action_spaces,
        discrete_actions,
        gamma,
        horizon=None,
        dt=1e-1,
        has_obs=False,
        has_action_masks=False,
        n_agents=1,
    ):
        """
            Initializes an environment object with specific state space, observation space,
            action space, and other parameters necessary for defining the environment's
            dynamics, decision-making properties, and agent specifications.

            Attributes:
            state_space (Space): The state space of the environment, defining all possible
                states the environment can be in.
            observation_spaces (Space): The observation spaces of the environment, representing
                all valid observations different agents can perceive.
            action_space (Space): The action spaces of the environment, detailing all actions
                different agents can take.
            discrete_actions (bool): Whether actions in the environment are discrete.
            gamma (float): Discount factor used for reward computations.
            horizon (int or None): Optional maximum number of steps for each episode.
            dt (float): Time delta representing the step interval for the environment updates.
            has_obs (bool): Indicates if the environment provides observations.
            has_action_masks (bool): Specifies whether the environment supports action masks,
                enabling/limiting viable actions at any time step.
            n_agents (int): Number of agents in the environment.
        """
        self.state_space = state_space
        self._observation_spaces = observation_spaces
        self.observation_space = observation_spaces[0]
        self._action_spaces = action_spaces
        self.action_space = action_spaces[0]
        self.discrete_actions = discrete_actions
        self.gamma = gamma
        self.horizon = horizon
        self.dt = dt
        self.has_obs = has_obs
        self.has_action_masks = has_action_masks
        self.n_agents = n_agents

        self._add_save_attr(
            state_space="mushroom",
            observation_space="mushroom",
            action_space="mushroom",
            gamma="primitive",
            horizon="primitive",
            dt="primitive",
            has_obs="primitive",
            has_action_masks="primitive",
            n_agents="primitive",
        )

    def action_space_for_idx(self, agent_index):
        return self._pick_space(self._action_spaces, agent_index)

    def observation_space_for_idx(self, agent_index):
        return self._pick_space(self._observation_spaces, agent_index)

    def _pick_space(self, spaces, agent_index):
        if agent_index >= self.n_agents or agent_index < -1:
            raise ValueError("Agent index out of range.")
        if self.n_agents > len(spaces) or agent_index == -1:
            assert len(spaces) == 1
            return spaces[0]
        return spaces[agent_index]
