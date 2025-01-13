# import
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from Mushroom.environments.fluid.circular_network import fmu_paths, model_classes, connections_config, \
    parameters_to_log, CircularFluidNetwork
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

speeds = np.array([0.45, 0.45])
c = 0
while not sim.is_done():
    sim.do_simulation_step(speeds)

results = sim.get_results()

print(f"Min power consumption of pump 4: {results['water_network.P_pum_4'].min()}")
print(f"Max power consumption of pump 4: {results['water_network.P_pum_4'].max()}")
print(f"Max flow through pump 4: {results['water_network.V_flow_4'].max()}")
print(f"Min power consumption of pump 1: {results['water_network.P_pum_1'].min()}")
print(f"Max power consumption of pump 1: {results['water_network.P_pum_1'].max()}")
print(f"Max flow through pump 1: {results['water_network.V_flow_1'].max()}")

print(f"Max flow through valves: 2: "
      f"{results['water_network.V_flow_2'].max()}, "
      f"3: {results['water_network.V_flow_3'].max()}, "
      f"5: {results['water_network.V_flow_5'].max()}, "
      f"6: {results['water_network.V_flow_6'].max()}"
      )

print(f"Max flow sum: {results['water_network.V_flow_2'].max() + results['water_network.V_flow_3'].max() + results['water_network.V_flow_5'].max() + results['water_network.V_flow_6'].max()}")

valves = [2, 3, 5, 6]
pumps = [1, 4]

CircularFluidNetwork.plot_valve_and_pump_data(
    time=results["time"],
    valves=valves,
    valve_openings=[results[f"water_network.u_v_{v}"] for v in valves],
    valve_demands=[results[f"control_api.demand_v_{v}"] for v in valves],
    valve_flows=[results[f"water_network.V_flow_{v}"] for v in valves],
    pumps=pumps,
    pump_speeds=[results[f"control_api.w_p_{p}"] for p in pumps],
    pump_powers=[results[f"water_network.P_pum_{p}"] for p in pumps],
    pump_flows=[results[f"water_network.V_flow_{p}"] for p in pumps],
    title="/",
)
quit(0)

fig, ax = plt.subplots()
ax.plot(
    results["time"],
    results["water_network.V_flow_2"],
    lw=1.5,
    label="actual flow",
    marker=".",
    markersize=8,
)
ax.plot(
    results["time"],
    results["control_api.w_v_2"],
    lw=2,
    label="demand",
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground=[77 / 255, 73 / 255, 67 / 255]),
        pe.Normal(),
    ],
)
ax.set_xlabel("TIME in s")
ax.set_ylabel("VOLUME FLOW in m$^3$/h")
ax.set_title("VALVE 2")
ax.legend(loc="upper left", ncol=2)
fig.show()

# display results - consumer 6
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax2.plot(results["time"], results["water_network.u_v_6"], lw=1.5)

ax.plot(
    results["time"],
    results["control_api.w_v_6"],
    lw=1.5,
    label="demand",
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground=[77 / 255, 73 / 255, 67 / 255]),
        pe.Normal(),
    ],
    c=[253 / 255, 202 / 255, 0 / 255],
)
ax.plot(
    results["time"],
    results["water_network.V_flow_6"],
    lw=1.5,
    label="actual flow",
    markersize=8,
    c=[233 / 255, 80 / 255, 62 / 255],
)
ax.set_xlabel("TIME in s")
ax.set_ylabel("VOLUME FLOW in m$^3$/h")
ax2.set_ylabel("VALVE POSITION in %", c=[0 / 255, 78 / 255, 115 / 255])
ax.set_title("VALVE 6")
ax.legend(loc="lower right", ncol=2)
ax2.spines["right"].set_visible(True)
fig.show()

# display results - pump 4
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.plot(results["time"], results["control_api.w_p_4"], lw=1.5)
ax1.plot(
    results["time"],
    results["water_network.V_flow_4"],
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground=[77 / 255, 73 / 255, 67 / 255]),
        pe.Normal(),
    ],
    c=[253 / 255, 202 / 255, 0 / 255],
    label="actual flow",
)
ax1.set_xlabel("TIME in s")
ax2.set_ylabel("ROTATIONAL SPEED in rpm", c=[0 / 255, 78 / 255, 115 / 255])
ax2.spines["right"].set_visible(True)
ax1.set_ylabel("VOLUME FLOW in m$^3$/h ")
ax1.set_title("PUMP 4")
fig.show()

# display results - pump 1
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.plot(results["time"], results["control_api.w_p_1"], lw=1.5)
ax1.plot(
    results["time"],
    results["water_network.V_flow_1"],
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground=[77 / 255, 73 / 255, 67 / 255]),
        pe.Normal(),
    ],
    c=[253 / 255, 202 / 255, 0 / 255],
    lw=1.5,
    label="actual flow",
)
ax1.set_xlabel("TIME in s")
ax2.set_ylabel("ROTATIONAL SPEED in rpm", c=[0 / 255, 78 / 255, 115 / 255])
ax2.spines["right"].set_visible(True)
ax1.set_ylabel("VOLUME FLOW in m$^3$/h ")
ax1.set_title("PUMP 1")
fig.show()

# display results - valve 2, 3, & 5
fig, ax = plt.subplots()
ax.plot(results["time"], results["water_network.u_v_2"] * 100, label="valve 2")
ax.plot(results["time"], results["water_network.u_v_3"] * 100, label="valve 3")
ax.plot(results["time"], results["water_network.u_v_5"] * 100, label="valve 4")
ax.set_xlabel("TIME in s")
ax.set_ylabel("VALVE POSITION in %")
ax.legend()
ax.set_title("VALVE 2, 3 & 5")
fig.show()
