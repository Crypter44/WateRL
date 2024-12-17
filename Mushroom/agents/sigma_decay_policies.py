import numpy as np
from mushroom_rl.policy import GaussianPolicy
from mushroom_rl.policy.noise_policy import OrnsteinUhlenbeckPolicy
from scipy.stats import stats


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
    def __init__(self, mu, initial_sigma, target_sigma, updates_till_target_reached):
        super().__init__(mu, np.array([[float(initial_sigma)]]))
        self._sigma = float(initial_sigma)
        self._inv_sigma = 1 / self._sigma

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
            self._inv_sigma = 1 / self._sigma

    def get_sigma(self):
        return self._sigma

    def set_sigma(self, sigma):
        self._sigma = float(sigma)
        self._inv_sigma = 1 / sigma

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
    if active:
        for a in [agents]:
            a.policy.reactivate_noise()
    else:
        for a in [agents]:
            a.policy.deactivate_noise()


def update_sigma_for_all(agents):
    for a in [agents]:
        a.policy.update_sigma()
