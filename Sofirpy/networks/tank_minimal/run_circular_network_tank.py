import numpy as np
from tqdm import tqdm

from Mushroom.environments.fluid.minimal_tank_network import MinimalTankNetwork
from Sofirpy.networks.tank_minimal.config import get_minimal_tank_network_config
from Sofirpy.simulation import ManualStepSimulator

stop_time = 55_000.0

sim = ManualStepSimulator(
    stop_time=stop_time,
    step_size=1.0,
    **get_minimal_tank_network_config(),
    start_values={"water_network": {"tank_9.crossArea": 1.25, "tank_9.height": 2}},
    logging_step_size=1,
    verbose=False,
    ignore_warnings=False,
)
sim.reset_simulation(
    stop_time=stop_time,
    step_size=1.0,
)
for time in tqdm(range(int(stop_time))):
    if time < 8_000:
        action = np.array([1.0, 0.0])
    elif time < 45_000:
        action = np.array([1.0, 0.0])
    elif time > 45_000:
        action = np.array([1.0, 1.0])
    try:
        sim.do_simulation_step(action)
    except Exception as e:
        print("-------- Caught exception --------")
        print(e)
        break

MinimalTankNetwork._render_task(sim.get_results())
sim.finalize()