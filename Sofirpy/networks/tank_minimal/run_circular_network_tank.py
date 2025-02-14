import numpy as np
from tqdm import tqdm

from Mushroom.environments.fluid.minimal_tank_network import MinimalTankNetwork
from Sofirpy.networks.tank_minimal.config import get_minimal_tank_network_config
from Sofirpy.simulation import ManualStepSimulator

stop_time = 86_400.0

sim = ManualStepSimulator(
    stop_time=stop_time,
    step_size=20.0,
    **get_minimal_tank_network_config("tagesgang"),
    logging_step_size=20.0,
    verbose=False,
    ignore_warnings=False,
)
sim.reset_simulation(
    stop_time=stop_time,
    step_size=20.0,
    logging_step_size=20.0,
)

pbar = tqdm(total=len(sim._time_series))
while not sim.is_done():
    demand = sim.get_current_state()[0]["control_api"][0]
    v = 0.0
    if demand < 1.45:
        v = 1.0
    if demand > 3.68:
        v = 1.0

    action = np.array([1.0, v])
    try:
        sim.do_simulation_step(action)
    except Exception as e:
        print("-------- Caught exception --------")
        print(e)
        break
    pbar.update(1)
pbar.close()

MinimalTankNetwork._render_task(sim.get_results())

print(max(sim.get_results()["water_network.u_v_5"]))
sim.finalize()
