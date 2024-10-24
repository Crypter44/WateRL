# %% import
import json
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path
import time

from sofirpy import simulate

from control_api import ControlAPI

dir_path = Path(__file__).parent
plt.style.use(dir_path / "FST.mplstyle")

# %%

omega = 2.65711342e-01
time_demand_curve = np.linspace(0, 24, 100)
demand_volume_flow_m3h = (
    10  # daily demand in m3
    # the following function gives a demand per time unit in % of the daily demand
    # the total daily demand is reached after 24 time units
    * (
        4.21452027e-02
        - 1.41579420e-02 * np.cos(omega * time_demand_curve)
        - 1.62752679e-02 * np.sin(omega * time_demand_curve)
        + 5.94972876e-03 * np.cos(2 * omega * time_demand_curve)
        - 1.82545802e-02 * np.sin(2 * omega * time_demand_curve)
        - 2.72810544e-03 * np.cos(3 * omega * time_demand_curve)
        + 2.15704832e-03 * np.sin(3 * omega * time_demand_curve)
        - 5.15308835e-03 * np.cos(4 * omega * time_demand_curve)
        - 1.10491878e-04 * np.sin(4 * omega * time_demand_curve)
    )
)
plt.plot(time_demand_curve, demand_volume_flow_m3h)

# %% setup simulation

fmu_dir_path = dir_path.parent / "Fluid_Model"

fmu_path = fmu_dir_path / "mini_circular_water_network.fmu"

agent_config_path = fmu_dir_path / "mini_circular_water_network.json"

connections_config_path = (
    fmu_dir_path / "mini_circular_water_network_connections_config.json"
)

logging_config_path = (
    fmu_dir_path / "mini_circular_water_network_parameters_to_log.json"
)

# create interface of multi-agent system to FMU
model_classes = {"control_api": ControlAPI}
fmu_paths = {"water_network": str(fmu_path)}

with open(connections_config_path) as connections_config_json:
    connections_config = json.load(connections_config_json)

with open(logging_config_path) as logging_config_json:
    parameters_to_log = json.load(logging_config_json)

# %% run simulation
start_time = time.time()
results, units = simulate(
    stop_time=50,
    step_size=1,
    fmu_paths=fmu_paths,
    model_classes=model_classes,
    connections_config=connections_config,
    parameters_to_log=parameters_to_log,
    logging_step_size=1,
    get_units=True,
)

# %% display results - volume flow, consumer 2

fig, ax = plt.subplots()
ax.plot(
    results["time"],
    results["water_network.V_flow_7"],
    lw=1.5,
    label="actual flow",
    marker=".",
    markersize=8,
)
ax.plot(
    results["time"],
    results["control_api.w_v_7"],
    lw=2,
    label="demand",
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground=[77 / 255, 73 / 255, 67 / 255]),
        pe.Normal(),
    ],
)
ax.set_xlabel("TIME in s")
ax.set_ylabel("VOLUME FLOW in m$^3$/h")
ax.set_title("VALVE 6")
ax.legend(loc="lower right", ncol=2)


# %% display results - volume flow, consumer 3
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax2.plot(results["time"], results["water_network.u_v_7"], lw=1.5)

ax.plot(
    results["time"],
    results["control_api.w_v_7"],
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
    results["water_network.V_flow_7"],
    lw=1.5,
    label="actual flow",
    markersize=8,
    c=[233 / 255, 80 / 255, 62 / 255],
)
ax.set_xlabel("TIME in s")
ax.set_ylabel("VOLUME FLOW in m$^3$/h")
ax2.set_ylabel("VALVE POSITION in %")
ax.set_title("VALVE 2")
ax.legend(loc="lower right", ncol=2)
ax2.spines["right"].set_visible(True)
# %%
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.plot(results["time"], results["water_network.u_p_4"], lw=1.5)
ax1.plot(
    results["time"],
    results["control_api.w_p_4"],
    c=[253 / 255, 202 / 255, 0 / 255],
    lw=1.5,
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground=[77 / 255, 73 / 255, 67 / 255]),
        pe.Normal(),
    ],
    label="flow @ PI-Controller",
    # visible=False
)
ax1.plot(
    results["time"],
    results["water_network.V_flow_4"],
    c=[233 / 255, 80 / 255, 62 / 255],
    lw=1.5,
    label="actual flow",
    # visible=False
)
ax1.set_xlabel("TIME in s")
ax2.set_ylabel("ROTATIONAL SPEED in rpm", c=[0 / 255, 78 / 255, 115 / 255])
ax2.spines["right"].set_visible(True)
ax1.set_ylabel("VOLUME FLOW in m$^3$/h ")
ax1.legend(loc="lower right", ncol=2)
ax1.set_title("PUMP 6")
# %%
fig, ax = plt.subplots()
ax.plot(results["time"], results["water_network.u_v_2"] * 100, label="valve 2")
ax.plot(results["time"], results["water_network.u_v_3"] * 100, label="valve 3")
# ax.plot(results["time"], results["water_network.u_v_4"] * 100, label="valve 4")
ax.set_xlabel("TIME in s")
ax.set_ylabel("VALVE POSITION in %")
ax.legend()
# ax.set_ylim([-0.1, 100])
# ax.set_title("VALVE 2, 3 & 4")
ax.set_title("VALVE 2 & 3 ")
