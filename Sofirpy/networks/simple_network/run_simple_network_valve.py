# %% import

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from Sofirpy.networks.simple_network.config import get_simple_network_valve_config
from Sofirpy.simulation import ManualStepSimulator

# create simulation
sim = ManualStepSimulator(
    stop_time=100,
    step_size=1,
    config=get_simple_network_valve_config(),
    logging_step_size=1,
    get_units=True,
    verbose=True,
)

# Run the simulation
out = 0
while not sim.is_done():
    # calculate the control action
    error_flow = sim.systems["control_api"].simulation_entity.requested_volume_flow - sim.systems[
        "water_network"].simulation_entity.get_parameter_value("V_flow_2")
    u = out + 0.1 * error_flow

    # limitation of actions
    if u > 1:
        u = 1
    if u < 0:
        u = 0

    out = u

    # do simulation step with the calculated action
    sim.do_simulation_step(np.array([u]))

# finalize simulation and get results
results, _ = sim.get_results()
sim.finalize()

# %% display results - consumer 6
fig, ax = plt.subplots(figsize=(10, 8))
ax2 = ax.twinx()

ax.plot(
    np.linspace(0, 100, 100),
    [sim.systems["control_api"].simulation_entity.requested_volume_flow] * 100,
    lw=1.5,
    label="DEMAND",
    linestyle=(0, (2, 1)),
    c=[0 / 255, 78 / 255, 115 / 255],
)
ax.plot(
    results["time"],
    results["water_network.V_flow_2"],
    lw=1.5,
    label="VOLUME FLOW",
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground=[77 / 255, 73 / 255, 67 / 255]),
        pe.Normal(),
    ],
    c=[253 / 255, 202 / 255, 0 / 255],
)
ax2.plot(
    results["time"],
    results["control_api.w_v_2"],
    lw=1.5,
    label="VALVE OPENING",
    markersize=8,
    c=[0 / 255, 78 / 255, 115 / 255],
)
ax.set_xlabel("TIME in s")
ax.set_ylabel("VOLUME FLOW AT VALVE in m$^3$/h")
ax2.set_ylabel("VALVE OPENING in %", c=[0 / 255, 78 / 255, 115 / 255])
ax2.spines["right"].set_visible(True)
fig.legend()
fig.show()
