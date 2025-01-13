import warnings

import numpy as np
from mushroom_rl.policy import GaussianPolicy
from mushroom_rl.policy.noise_policy import OrnsteinUhlenbeckPolicy
from scipy.stats import stats


class Decay:
    def __init__(self, checkpoints, decay_type='linear'):
        self.checkpoints = checkpoints
        self.next_checkpoint = checkpoints.pop(0)
        self.current = self.next_checkpoint[1]
        self.num_updates = 0
        self.decay_type = decay_type

    def decay(self):
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
        self.current += (self.next_checkpoint[1] - self.current) / self._updates_till_next_checkpoint()

    def _decay_exponential(self):
        self.current *= (self.next_checkpoint[1] / self.current) ** (1 / self._updates_till_next_checkpoint())

    def _updates_till_next_checkpoint(self):
        return self.next_checkpoint[0] - self.num_updates + 1

    def get(self):
        return self.current


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
    def __init__(
            self,
            mu,
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
    # Ensure agents is always iterable
    for agent in (agents if hasattr(agents, '__iter__') and not isinstance(agents, (str, bytes)) else [agents]):
        if active:
            agent.policy.reactivate_noise()
        else:
            agent.policy.deactivate_noise()


def update_sigma_for_all(agents):
    # Ensure agents is always iterable
    for agent in (agents if hasattr(agents, '__iter__') and not isinstance(agents, (str, bytes)) else [agents]):
        agent.policy.update_sigma()
