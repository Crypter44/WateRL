from mushroom_rl.environments import GridWorld
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Core
import numpy as np

mdp = GridWorld(width=3, height=3, goal=(2, 2), start=(0, 0))

epsilon = Parameter(value=.1)
policy = EpsGreedy(epsilon=epsilon)

learning_rate = Parameter(value=.6)
agent = QLearning(mdp.info, policy, learning_rate)

core = Core(agent, mdp)
core.learn(n_steps=100000, n_steps_per_fit=1, render=False)

core.evaluate(n_episodes=10, render=True)

shape = agent.Q.shape
q = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        state = np.array([i])
        action = np.array([j])
        q[i, j] = agent.Q.predict(state, action)
print(q)