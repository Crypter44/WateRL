# %% import
import json
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path
import time

from sofirpy import simulate, SimulationEntity

dir_path = Path(__file__).parent
# plt.style.use(dir_path / "FST.mplstyle")

# %% setup simulation

fmu_dir_path = dir_path.parent.parent / "fluid_models" / "mini_water_network_tank"
fmu_path = fmu_dir_path / "mini_tank.fmu"

connections_config_path = fmu_dir_path / "mini_tank_connections_config.json"
with open(connections_config_path) as connections_config_json:
    connections_config = json.load(connections_config_json)

logging_config_path = fmu_dir_path / "mini_tank_parameters_to_log.json"
with open(logging_config_path) as logging_config_json:
    parameters_to_log = json.load(logging_config_json)


# %%
class Controler(SimulationEntity):
    """This Class is used when generating the input values for the FMU.

    It connects the input and output values for the FMU to a custom code.
    """

    def __init__(self) -> None:
        self.inputs = {
            "V_flow_4": 0.0,
            "p_rel_4": 0.0,
            "P_pum_4": 0.0,
            "V_flow_5": 0.0,
            "p_rel_5": 0.0,
            "u_v_5": 0.0,
            "V_flow_7": 0.0,
            "p_rel_7": 0.0,
            "level_tank_9": 0.0,
        }
        self.outputs = {
            "w_p_4": 0.0,
            "w_v_5": 0.0,
            "w_v_7": 0.0,
        }

    def do_step(self, time: float):  # mandatory method
        """This code is executed during each simulation step.

        Args:
            time (float): Simulated timestep

        """
        if time < 10000:
            # Bedarf ist so gering, dass Tank befüllt werden kann
            self.outputs["w_p_4"] = 1
            self.outputs["w_v_5"] = 0.4
            self.outputs["w_v_7"] = 0
        elif time < 20000:
            self.outputs["w_p_4"] = 1
            self.outputs["w_v_5"] = 0.4
            self.outputs["w_v_7"] = 1
        elif time < 30000:
            self.outputs["w_p_4"] = 1
            self.outputs["w_v_5"] = 3
            self.outputs["w_v_7"] = 0
        elif time > 30000:
            self.outputs["w_p_4"] = 1
            self.outputs["w_v_5"] = 3
            self.outputs["w_v_7"] = 1

    def set_parameter(
        self, parameter_name: str, parameter_value: float
    ):  # mandatory method
        """Gets parameters from the FMU.

        Args:
            parameter_name (str): Name of the value as given in the connections_config.
            parameter_value (float): Value of the parameter.
        """
        self.inputs[parameter_name] = parameter_value

    def get_parameter_value(self, output_name: str) -> float:  # mandatory method
        """Extracts parameters that are imposed on the FMU.

        Args:
            output_name (str): Name of the value as given in the connections_config.

        Returns:
            float: Value of the parameter.
        """
        return self.outputs[output_name]

    def conclude_simulation(self):  # optional
        """Just to make sure."""
        print("Concluded simulation!")


# %% run simulation

# create interface of multi-agent system to FMU
model_classes = {"control_api": Controler}
fmu_paths = {"water_network": str(fmu_path)}

start_time = time.time()
results, units = simulate(
    stop_time=50000.0,
    step_size=10.0,
    fmu_paths=fmu_paths,
    model_classes=model_classes,
    connections_config=connections_config,
    parameters_to_log=parameters_to_log,
    logging_step_size=10,
    get_units=True,
)

# %% display results - pump 4
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.plot(
    results["time"],
    results["water_network.V_flow_4"],
    lw=1.5,
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground=[77 / 255, 73 / 255, 67 / 255]),
        pe.Normal(),
    ],
    c=[253 / 255, 202 / 255, 0 / 255],
)
ax2.plot(
    results["time"],
    results["water_network.P_pum_4"],
    lw=1.5,
    markersize=8,
    c=[0 / 255, 78 / 255, 115 / 255],
)
ax.set_xlabel("TIME in s")
ax.set_ylabel("VOLUME FLOW in m$^3$/h")
ax2.set_ylabel("PUMP POWER in W", c=[0 / 255, 78 / 255, 115 / 255])
ax2.spines["right"].set_visible(True)
ax.set_title("PUMP 4")
fig.show()
# %% display results - valve 5
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.plot(
    results["time"],
    results["water_network.V_flow_5"],
    lw=1.5,
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground=[77 / 255, 73 / 255, 67 / 255]),
        pe.Normal(),
    ],
    c=[253 / 255, 202 / 255, 0 / 255],
)
ax2.plot(
    results["time"],
    results["water_network.u_v_5"],
    lw=1.5,
    markersize=8,
    c=[0 / 255, 78 / 255, 115 / 255],
)
ax.set_xlabel("TIME in s")
ax.set_ylabel("VOLUME FLOW in m$^3$/h")
ax2.set_ylabel("OPENING RATE", c=[0 / 255, 78 / 255, 115 / 255])
ax2.spines["right"].set_visible(True)
ax.set_title("VALVE 5")
fig.show()
# %% display results - tank 9
fig, ax = plt.subplots()
ax.plot(
    results["time"],
    results["water_network.V_flow_7"],
    lw=1.5,
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground=[77 / 255, 73 / 255, 67 / 255]),
        pe.Normal(),
    ],
    c=[253 / 255, 202 / 255, 0 / 255],
)
ax2 = ax.twinx()
ax.set_xlabel("TIME in s")
ax.set_ylabel("VOLUME FLOW in m$^3$/h")

ax2.plot(
    results["time"],
    results["water_network.level_tank_9"],
    lw=1.5,
    c=[0 / 255, 78 / 255, 115 / 255],
)
ax2.set_ylabel("TANK LEVEL in m", c=[0 / 255, 78 / 255, 115 / 255])
ax2.set_title("TANK")
fig.show()
# %%
fig, ax = plt.subplots()
ax.plot(
    results["time"],
    results["water_network.p_rel_7"],
    lw=1.5,
    c=[0 / 255, 78 / 255, 115 / 255],
)
ax.hlines(y=0, xmin=0, xmax=50000, colors="k", linestyles=":")
ax.set_xlabel("TIME in s")
ax.set_ylabel("RELATIVE PRESSURE @ TANK-VALVE in bar")
ax.set_title("TANK-VALVE")
fig.show()
# %%
