import numpy as np
from tqdm import tqdm

from Mushroom.environments.fluid.minimal_tank_network import MinimalTankNetwork
from Sofirpy.networks.tank_minimal.config import get_minimal_tank_network_config
from Sofirpy.simulation import ManualStepSimulator

stop_time = 86_400.0

sim = ManualStepSimulator(
    stop_time=stop_time,
    step_size=20.0,
    **get_minimal_tank_network_config("tagesgang", exclude_start_values=True),
    start_values={
        "water_network": {
            "tank_9.crossArea": 3,
            "tank_9.height": 3,
            "init_level_tank_9": 0.05,
            "elevation_tank_9": 14.5,
        }
    },
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
        v = 0.0
    if demand > 3.58:
        v = 0.0

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

print(f"max opening {max(sim.get_results()['water_network.u_v_5'])}")

demand = np.array(sim.get_results()["control_api.w_v_5"])[2:]
flow = np.array(sim.get_results()["water_network.V_flow_5"])[2:]

diff = demand - flow
print(f"max diff {max(np.abs(diff))}")
print(f"max diff at {np.argmax(np.abs(diff)) / len(diff)} %")
print(len(diff))

# plot diff
import matplotlib.pyplot as plt

plt.plot(diff)
plt.show()

sim.finalize()
