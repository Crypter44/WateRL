import warnings
from copy import deepcopy

import numpy as np
from mushroom_rl.policy import GaussianPolicy
from mushroom_rl.policy.noise_policy import OrnsteinUhlenbeckPolicy
from scipy.stats import stats


class Decay:
    """
    Represents a decay system for adjusting values based on defined checkpoints.

    The Decay class allows gradual transition of a parameter's value through a series
    of checkpoints over time. It supports linear and exponential decay types, providing
    flexibility for various use cases in parameter scheduling. The class maintains the
    current value, the next checkpoint, and tracks the number of updates performed.

    Attributes:
        checkpoints (list[tuple[int, float]]): A list of checkpoints, where each checkpoint
            is a tuple containing the number of updates at which the checkpoint occurs and
            the value to be reached at that checkpoint.
        next_checkpoint (tuple[int, float]): The next checkpoint to transition to, updated
            dynamically as checkpoints are completed.
        current (float): The current value of the parameter, updated based on the decay logic.
        num_updates (int): The number of updates that have occurred so far.
        decay_type (str): The type of decay to use, either 'linear' or 'exponential'.
    """
    def __init__(self, checkpoints, decay_type='linear'):
        self.checkpoints = deepcopy(checkpoints)
        self.next_checkpoint = self.checkpoints.pop(0)
        self.current = self.next_checkpoint[1]
        self.num_updates = 0
        self.decay_type = decay_type

    def decay(self):
        """
        Controls the decay behavior for a variable dependent on checkpoints and decay type.

        The method manages updates of the internal value `current` based on predefined
        checkpoints and the decay type. It either applies linear or exponential decay upon
        certain conditions and raises an error if an invalid decay type is encountered.

        Raises
        ------
        ValueError
            If the decay type is not 'linear' or 'exponential'.

        Returns
        -------
        float
            The current updated value after applying the appropriate decay logic.
        """
        if self.num_updates == self.next_checkpoint[0]:
            self.current = self.next_checkpoint[1]
            if len(self.checkpoints) > 0:
                self.next_checkpoint = self.checkpoints.pop(0)
        else:
            if self.next_checkpoint[1] != self.current:
                if self.decay_type == 'linear':
                    self._decay_linear()
                elif self.decay_type == 'exponential':
                    self._decay_exponential()
                else:
                    raise ValueError('Invalid decay type')
        self.num_updates += 1
        return self.current

    def _decay_linear(self):
        """
        Applies linear decay to the current value based on the next checkpoint.
        """
        self.current += (self.next_checkpoint[1] - self.current) / self._updates_till_next_checkpoint()

    def _decay_exponential(self):
        """
        Applies exponential decay to the current value based on the next checkpoint.
        """
        self.current *= (self.next_checkpoint[1] / self.current) ** (1 / self._updates_till_next_checkpoint())

    def _updates_till_next_checkpoint(self):
        return self.next_checkpoint[0] - self.num_updates + 1

    def get(self):
        return self.current

    def skip_to_next_checkpoint(self):
        """
        Skips to the next checkpoint in the decay process.
        """
        self.current = self.next_checkpoint[1]
        if len(self.checkpoints) > 0:
            self.next_checkpoint = self.checkpoints.pop(0)
        self.num_updates = self.next_checkpoint[0]


class OUPolicyWithNoiseDecay(OrnsteinUhlenbeckPolicy):
    def __init__(self, mu, initial_sigma, target_sigma, updates_till_target_reached, theta, dt):
        super().__init__(mu, initial_sigma, theta, dt)
        self._initial_sigma = initial_sigma
        self._target_sigma = target_sigma
        self._updates_till_target_reached = updates_till_target_reached
        self._sigma_tmp = initial_sigma

    def deactivate_noise(self):
        self._sigma_tmp = self._sigma
        self._sigma = 0

    def reactivate_noise(self):
        self._sigma = self._sigma_tmp

    def update_sigma(self):
        if self._sigma != self._target_sigma:
            self._sigma *= (self._target_sigma / self._initial_sigma) ** (1 / self._updates_till_target_reached)


class UnivariateGaussianPolicy(GaussianPolicy):
    """
    Represents a univariate Gaussian policy for action sampling in reinforcement learning.

    This class defines a univariate Gaussian policy.
    It allows for dynamic adjustments of the variance based on a specified decay schedule
    or provided checkpoints.

    Attributes:
        _sigma : float
            Current standard deviation of the Gaussian policy.
        _inv_sigma : float
            Inverse of the current standard deviation for computational efficiency.
        _sigma_decay : Decay
            An instance of the Decay class handling the decay behavior of the variance.
        _sigma_tmp : float
            A temporary storage for the latest active sigma, used in noise activation/deactivation.
        _approximator : Any
            A function or model used to approximate the mean value (mu) based on the state.
    """
    def __init__(
            self,
            mu=None,
            sigma_checkpoints=None,
            decay_type='exponential',
            initial_sigma=None,
            target_sigma=None,
            updates_till_target_reached=None
    ):
        if sigma_checkpoints is None:
            assert (
                    initial_sigma is not None
                    and target_sigma is not None
                    and updates_till_target_reached is not None
            ), 'Either sigma_checkpoints or initial_sigma, target_sigma and updates_till_target_reached must be given'
            sigma_checkpoints = [(0, initial_sigma), (updates_till_target_reached, target_sigma)]
        else:
            if initial_sigma is not None or target_sigma is not None or updates_till_target_reached is not None:
                warnings.warn(
                    'sigma_checkpoints given, ignoring initial_sigma, target_sigma and updates_till_target_reached'
                )

        super().__init__(mu, np.array([[float(sigma_checkpoints[0][1])]]))
        self._inv_sigma = 1 / self._sigma
        self._sigma_decay = Decay(sigma_checkpoints, decay_type)
        self._sigma = self._sigma_decay.get()
        self._sigma_tmp = self._sigma

    def deactivate_noise(self):
        self._sigma_tmp = self._sigma
        self._sigma = 0

    def reactivate_noise(self):
        self._sigma = self._sigma_tmp

    def update_sigma(self):
        self._sigma = self._sigma_decay.decay()
        self._inv_sigma = 1 / self._sigma

    def skip_to_next_sigma(self):
        self._sigma_decay.skip_to_next_checkpoint()
        self._sigma = self._sigma_decay.get()
        self._inv_sigma = 1 / self._sigma

    def get_sigma(self):
        return self._sigma

    def set_sigma(self, sigma):
        self._sigma = float(sigma)
        self._inv_sigma = 1 / sigma

    def set_approximator(self, approximator):
        self._approximator = approximator

    def _compute_multivariate_gaussian(self, state):
        mu = np.reshape(self._approximator.predict(np.expand_dims(state, axis=0), **self._predict_params), -1)

        return mu, self._sigma, self._inv_sigma

    def __call__(self, state, action):
        mu, sigma = self._compute_multivariate_gaussian(state)[:2]

        return stats.norm.pdf(action, mu, sigma)

    def draw_action(self, state):
        mu, sigma = self._compute_multivariate_gaussian(state)[:2]

        return np.random.normal(mu, sigma)


def set_noise_for_all(agents, active):
    """
    Disable or enable noise for all agents.
    """
    # Ensure agents is always iterable
    for agent in (agents if hasattr(agents, '__iter__') and not isinstance(agents, (str, bytes)) else [agents]):
        if active:
            agent.policy.reactivate_noise()
        else:
            agent.policy.deactivate_noise()


def update_sigma_for_all(agents, sigma=None):
    """
    Update sigma for all agents.

    Based on the value of sigma, the sigma of the agents will be updated.
    If sigma is None, the sigma will be updated according to the decay policy.
    If sigma is a number, the sigma will be temporarily set to this value. After this, the decay policy can be continued by setting
    sigma to None and calling this function again.
    If sigma is "next" or "skip", the sigma will be set to the next checkpoint value or skipped to the next checkpoint.

    :param agents: list of agents or single agent
    :param sigma: float or None or "next" or "skip"
    """
    # Ensure agents is always iterable
    for agent in (agents if hasattr(agents, '__iter__') and not isinstance(agents, (str, bytes)) else [agents]):
        if sigma is None:
            agent.policy.update_sigma()
        elif type(sigma) in (int, float):
            agent.policy.set_sigma(sigma)
        elif sigma == "next" or sigma == "skip":
            agent.policy.skip_to_next_sigma()
        else:
            raise ValueError('Invalid sigma value! Must be a number or "next" or "skip" or None')

