import os

import numpy as np
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J
from tqdm import tqdm

from Mushroom.environments.fluid.simple_network_valve import SimpleNetworkValve
from Mushroom.utils.utils import set_seed

scores = {}

path = f"/home/js/PycharmProjects/Thesis/Mushroom/training/fluid/simple_network_valve/weights/"
mdp = SimpleNetworkValve(gamma=0.99, horizon=100)

files = os.listdir(path)

# files = ["agent_epo_60-s:1234-p1:0.003-p2:-200", "agent_epo_60-s:2345-p1:0.003-p2:-200"]

pbar = tqdm(files, unit='file')
for f in pbar:
    file = path + f

    agent = SAC.load(file)

    # Evaluate the agent
    set_seed(15)
    Js = []
    for i in range(100):
        core = Core(agent, mdp)
        dataset = core.evaluate(n_episodes=1, render=False, quiet=True)
        Js.append(compute_J(dataset, mdp.info.gamma))
        # core.mdp.render()

    score = np.mean(Js)
    scores[f] = score
    pbar.set_postfix(score=score)

sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
for k, v in sorted_scores.items():
    print(f"{k: <50}[{v:.4f}]")
