import numpy as np
from tqdm import tqdm

from Mushroom.environments.fluid.minimal_tank_network import MinimalTankNetwork
from Sofirpy.networks.tank_minimal.config import get_minimal_tank_network_config
from Sofirpy.simulation import ManualStepSimulator

sim = ManualStepSimulator(
    stop_time=50000.0,
    step_size=1.0,
    **get_minimal_tank_network_config(),
    logging_step_size=1,
    verbose=False,
    ignore_warnings=False,
)
sim.reset_simulation(
    stop_time=50000.0,
    step_size=1.0,
)
for time in tqdm(range(50000)):
    action = np.array([1.0, 1.0])
    try:
        sim.do_simulation_step(action)
    except Exception as e:
        print("-------- Caught exception --------")
        print(e)
        break

MinimalTankNetwork._render_task(sim.get_results())
sim.finalize()