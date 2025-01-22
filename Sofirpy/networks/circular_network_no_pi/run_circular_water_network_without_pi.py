# import
import numpy as np

from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Sofirpy.networks.circular_network_no_pi.config import get_circular_network_no_pi_config
from Sofirpy.simulation import ManualStepSimulator

sim = ManualStepSimulator(
    stop_time=200,
    step_size=1,
    config=get_circular_network_no_pi_config(),
    logging_step_size=1,
    get_units=False,
    verbose=True,
)

c = 0
while not sim.is_done():
    actions = np.array(
        [
            1,  # pump 1
            1,  # pump 4
            .252,  # valve 2
            0,  # valve 3
            0,  # valve 5
            0  # valve 6
        ]
    )
    sim.do_simulation_step(actions)
    c += 0.01
results = sim.get_results()

sim.finalize()

valves = [2, 3, 5, 6]
pumps = [1, 4]

CircularFluidNetwork.plot_valve_and_pump_data(
    results["time"],
    valves=valves,
    valve_openings=[results[f"control_api.w_v_{v}"] for v in valves],
    valve_demands=[results[f"control_api.demand_v_{v}"] for v in valves],
    valve_flows=[results[f"water_network.V_flow_{v}"] for v in valves],
    pumps=pumps,
    pump_speeds=[results[f"control_api.w_p_{p}"] for p in pumps],
    pump_powers=[results[f"water_network.V_flow_{p}"] for p in pumps],
    pump_flows=[results[f"water_network.V_flow_{p}"] for p in pumps],
)

# calculate the mean deviation of the valve flow from the demand for valve 2
valve_2_demand = results["control_api.demand_v_2"]
valve_2_flow = results["water_network.V_flow_2"]
mean_deviation = np.mean(valve_2_flow - valve_2_demand)
print(f"Mean deviation of valve 2 flow from demand: {mean_deviation}")
