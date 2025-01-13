import warnings

import numpy as np
from mushroom_rl.core import Serializable

from Mushroom.environments.mdp_info import MAMDPInfo


class MAEnvironment(Serializable):
    """
    Basic interface used by any multi-agent environment.

    """

    def __init__(self, mdp_info: MAMDPInfo):
        """
        Constructor.

        Args:
             mdp_info (MAMDPInfo): the information about the MDP.

        """
        self._mdp_info = mdp_info

        self._add_save_attr(_mdp_info="mushroom")

    def seed(self, seed):
        """
        Set the seed of the environment.

        Args:
            seed (float): the value of the seed.

        """
        if hasattr(self, 'env') and hasattr(self.env, 'seed'):
            self.env.seed(seed)
        else:
            warnings.warn('This environment has no custom seed. '
                          'The call will have no effect. '
                          'You can set the seed manually by setting numpy/torch seed')

    def reset(self):
        """
        Reset the current state.

        Returns:
            The current state.

        """
        raise NotImplementedError

    def step(self, actions):
        """
        Move the agent from its current state according to the action.

        Args:
            action, list[np.ndarray]: the list of actions to execute.

        Returns:
            The list of observations reached by the agent executing actions in their current
            state, the next state of the environment the rewards obtained in the transition and a flag to signal
            if the next state is absorbing. Also an additional info dictionary is
            returned (possibly empty).

        """
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def stop(self):
        """
        Method used to stop an mdp. Useful when dealing with real world
        environments, simulators, or when using openai-gym rendering

        """
        pass

    @property
    def info(self):
        """
        Returns:
             An object containing the info of the environment.

        """
        return self._mdp_info

    @staticmethod
    def _bound(x, min_value, max_value):
        """
        Method used to bound state and action variables.

        Args:
            x: the variable to bound;
            min_value: the minimum value;
            max_value: the maximum value;

        Returns:
            The bounded variable.

        """
        return np.maximum(min_value, np.minimum(x, max_value))
