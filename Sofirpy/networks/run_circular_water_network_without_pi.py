# import
import matplotlib.pyplot as plt
import numpy as np

from Mushroom.fluid_network_environments.circular_network_no_pi import fmu_paths, model_classes, connections_config, \
    parameters_to_log, connections_config_path
from Sofirpy.simulation import ManualStepSimulator

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

c = 0
while not sim.is_done():
    actions = np.array(
        [
            -0.1,  # pump 1
            -0.1,  # pump 4
            1.0,  # valve 2
            1.0,  # valve 3
            1.0,  # valve 5
            1.0  # valve 6
        ]
    )
    sim.do_simulation_step(actions)
    c += 0.01
results = sim.get_results()

sim.finalize()

# plot pump power
fig, ax = plt.subplots()
ax.plot(
    results["time"],
    results["water_network.P_pum_4"],
)
fig.show()

# plot pump rotational speed
fig, ax = plt.subplots()
ax.plot(
    results["time"],
    results["control_api.w_p_4"],
)
fig.show()

# plot pump volume flow
fig, ax = plt.subplots()
ax.plot(
    results["time"],
    results["water_network.V_flow_4"],
)
fig.show()
