import numpy as np
from tqdm import tqdm

from Mushroom.environments.fluid.minimal_tank_network import MinimalTankNetwork
from Sofirpy.networks.tank_minimal.config import get_minimal_tank_network_config
from Sofirpy.networks.tank_minimal.controller import ControllerMinimalTank
from Sofirpy.simulation import ManualStepSimulator

sim = ManualStepSimulator(
    stop_time=50000.0,
    step_size=1.0,
    **get_minimal_tank_network_config(),
    logging_step_size=1,
    verbose=False,
)

failed_counter = 0
test_count = 10

low = 0.0
high = 0.1

use_discrete_actions = False
use_random_initialization = True

for i in range(test_count):
    sim.reset_simulation(
        stop_time=50000.0,
        step_size=1.0,
    )
    if use_random_initialization:
        action = np.random.uniform(low, high, (2,))  # This is incredibly unstable, if high is set to anything above 0.1
        # It does not matter, whether we keep the action the same or wether we change it, the simulation will fail often
    else:
        action = np.zeros((2,))  # This is more stable, but still fails, if the action is changed too  much
    for time in tqdm(range(50000)):
        if time % ControllerMinimalTank.CONTROL_STEP_INTERVAL == 0:
            # This does not affect the stability
            # We model a normal pump agent with exploration noise
            action[0] = np.clip(np.random.normal(loc=action[0], scale=0.2), low, high)

            if use_discrete_actions:
                s = 0.001  # This is stable, at the cost of very low exploration
                action[1] += np.clip(s * np.random.choice([-1, +1]), 0, 1)
            else:
                no_change = 0  # Even no change in action will fail 6/10 times using random initialization
                low_scale = 0.01  # This scale works without random initialization, then 0/100 sims fail
                high_scale = 0.1  # This already fails 7/10 times without random initialization
                completely_random = 1.0  # This works, when the action is clipped to [0, 0.1]
                action[1] = np.clip(np.random.normal(loc=action[1], scale=completely_random), low, high)
        try:
            sim.do_simulation_step(action)
        except Exception as e:
            print("-------- Caught exception --------")
            failed_counter += 1
            print(e)
            break

    MinimalTankNetwork._render_task(sim.get_results())
sim.finalize()

print(f"Failed simulations: {failed_counter}/{test_count}")
