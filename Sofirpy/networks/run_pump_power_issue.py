import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from Mushroom.fluid_network_environments.circular_network import fmu_paths, model_classes, connections_config, \
    parameters_to_log
from Mushroom.utils import set_seed
from Sofirpy.simulation import ManualStepSimulator

set_seed(0)

sim = ManualStepSimulator(
    stop_time=200,
    step_size=1,
    fmu_paths=fmu_paths,
    model_classes=model_classes,
    connections_config=connections_config,
    parameters_to_log=parameters_to_log,
    logging_step_size=1,
    get_units=False,
    verbose=True,
)

speeds = np.array([0.9, 0.0])
c = 0
while not sim.is_done():
    if c == 2:
        speeds = np.array([0.0, 0.0])  # this seems to be the issue

        # UNCOMMENT THE FOLLOWING TO SEE THE PUMP WORKING
        # for the values I have tested > 0.02 the pump works fine, but for smaller values it does not
        # speeds = np.array([0.02, 0.0])
    else:
        speeds = np.array([0.9, 0.0])
    for i in range(10):
        sim.do_simulation_step(speeds)
    c += 1

results = sim.get_results()

fig, ax = plt.subplots()
ax.plot(
    results["time"],
    results["water_network.P_pum_1"],
    label="Pump 1 power",
)
ax.set_ylabel("Power [W]")
ax.set_title("PUMP POWER")
ax2 = ax.twinx()
ax2.plot(
    results["time"],
    results["control_api.w_p_1"],
    label="Pump 1 setpoint",
    linestyle="--",
)
ax2.set_ylabel("Setpoint")
fig.legend()
fig.show()

fig, ax = plt.subplots()
ax.plot(
    results["time"],
    results["water_network.V_flow_2"],
    lw=1.5,
    label="volume flow valve 2",
)
ax.set_title("VOLUME FLOW VALVE 2")
fig.show()

fig, ax = plt.subplots()
ax.plot(
    results["time"],
    results["water_network.V_flow_1"],
    lw=1.5,
    label="volume flow pump 1",
)
ax.plot(
    results["time"],
    results["water_network.V_flow_4"],
    lw=1.5,
    label="volume flow pump 4",
)
ax.set_title("VOLUME FLOW PUMPS")
fig.show()
