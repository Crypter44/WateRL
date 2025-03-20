import numpy as np
from tqdm import tqdm

from Mushroom.environments.fluid.minimal_tank_network import MinimalTankNetwork
from Sofirpy.networks.tank_minimal.config import get_minimal_tank_network_config
from Sofirpy.simulation import ManualStepSimulator

stop_time = 86_400.0
step_size = 20.0
use_logic = False

sim = ManualStepSimulator(
    stop_time=stop_time,
    step_size=step_size,
    **get_minimal_tank_network_config("tagesgang", exclude_start_values=True),
    start_values={
        "water_network": {
            "tank_9.crossArea": 3,
            "tank_9.height": 5,
            "init_level_tank_9": 0.05,
            "elevation_tank_9": 12.5,
        }
    },
    logging_step_size=step_size,
    verbose=False,
    ignore_warnings=False,
)
sim.reset_simulation(
    stop_time=stop_time,
    step_size=step_size,
    logging_step_size=step_size,
)

pbar = tqdm(total=len(sim._time_series))
while not sim.is_done():
    if use_logic:
        demand = sim.get_current_state()[0]["control_api"][0]
        s = demand ** 0.8 / 3.58 ** 0.8
        if demand < 0.7:
            s = 1 - (demand / 0.7) ** 2 + 0.5
        if demand > 3.58:
            s = 1.0

        s = np.clip(s, 0, 1)

        pressure_at_tank = sim.get_current_state()[0]["control_api"][9]
        v = 0
        if -0.22 < pressure_at_tank:
            v = 1.0

        action = np.array([s, v])
    else:
        action = np.array([1, 0])
    try:
        sim.do_simulation_step(action)
    except Exception as e:
        print("-------- Caught exception --------")
        print(e)
        break
    pbar.update(1)
pbar.close()


print(f"max opening {max(sim.get_results()['water_network.u_v_5'])}")

# calculate the total power consumption in kWh
power = sim.get_results()["water_network.P_pum_4"].sum() * step_size / 3600

print(f"Total power consumption: {np.round(power/1000, 4)} kWh")

# calculate power per flow
power_per_flow = (np.array(sim.get_results()["water_network.P_pum_4"])
                  / (np.array(sim.get_results()["water_network.V_flow_4"]) + 1))

print(f"Max power per flow: {np.round(power_per_flow.max(), 4)} W/m³/h")
print(f"Max power per flow at index: {power_per_flow.argmax()}")

print(f"Min power per flow: {np.round(power_per_flow.min(), 4)} W/m³/h")
print(f"Min power per flow at index: {power_per_flow.argmin()}")

MinimalTankNetwork._render_task(sim.get_results(), title=f"Power consumption: {np.round(power/1000, 4)} kWh")
sim.finalize()
