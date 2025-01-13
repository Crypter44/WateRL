# %% import
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
import time

from sofirpy import simulate, SimulationEntity

dir_path = Path(__file__).parent
# plt.style.use(dir_path / "FST.mplstyle")

# %% setup simulation

fmu_dir_path = dir_path.parent / "Fluid_Model" / "circular_water_network"
fmu_path = fmu_dir_path / "mini_circular_water_network.fmu"

connections_config = {
    "water_network": [
        {
            "parameter_name": "w_p_1",
            "connect_to_system": "control_api",
            "connect_to_external_parameter": "w_p_1",
        },
        {
            "parameter_name": "w_v_2",
            "connect_to_system": "control_api",
            "connect_to_external_parameter": "w_v_2",
        },
        {
            "parameter_name": "w_v_3",
            "connect_to_system": "control_api",
            "connect_to_external_parameter": "w_v_3",
        },
        {
            "parameter_name": "w_p_4",
            "connect_to_system": "control_api",
            "connect_to_external_parameter": "w_p_4",
        },
        {
            "parameter_name": "w_v_5",
            "connect_to_system": "control_api",
            "connect_to_external_parameter": "w_v_5",
        },
        {
            "parameter_name": "w_v_6",
            "connect_to_system": "control_api",
            "connect_to_external_parameter": "w_v_6",
        },
    ],
    "control_api": [
        {
            "parameter_name": "P_pum_1",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "P_pum_1",
        },
        {
            "parameter_name": "V_flow_1",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "V_flow_1",
        },
        {
            "parameter_name": "p_rel_1",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "p_rel_1",
        },
        {
            "parameter_name": "V_flow_2",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "V_flow_2",
        },
        {
            "parameter_name": "p_rel_2",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "p_rel_2",
        },
        {
            "parameter_name": "V_flow_3",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "V_flow_3",
        },
        {
            "parameter_name": "p_rel_3",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "p_rel_3",
        },
        {
            "parameter_name": "P_pum_4",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "P_pum_4",
        },
        {
            "parameter_name": "V_flow_4",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "V_flow_4",
        },
        {
            "parameter_name": "p_rel_4",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "p_rel_4",
        },
        {
            "parameter_name": "V_flow_5",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "V_flow_5",
        },
        {
            "parameter_name": "p_rel_5",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "p_rel_5",
        },
        {
            "parameter_name": "V_flow_6",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "V_flow_6",
        },
        {
            "parameter_name": "p_rel_6",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "p_rel_6",
        },
    ],
}

parameters_to_log = {
    "water_network": [
        "V_flow_1",
        "p_rel_1",
        "P_pum_1",
        "V_flow_2",
        "p_rel_2",
        "V_flow_3",
        "p_rel_3",
        "V_flow_4",
        "p_rel_4",
        "P_pum_4",
        "V_flow_5",
        "p_rel_5",
        "V_flow_6",
        "p_rel_6",
    ],
    "control_api": ["w_p_1", "w_v_2", "w_v_3", "w_p_4", "w_v_5", "w_v_6"],
}


# %%
class Controler(SimulationEntity):
    """This Class is used when generating the input values for the FMU.

    It connects the input and output values for the FMU to a custom code.
    """

    def __init__(self) -> None:
        self.inputs = {
            "V_flow_1": 0.0,
            "p_rel_1": 0.0,
            "P_pum_1": 0.0,
            "V_flow_2": 0.0,
            "p_rel_2": 0.0,
            "V_flow_3": 0.0,
            "p_rel_3": 0.0,
            "V_flow_4": 0.0,
            "p_rel_4": 0.0,
            "P_pum_4": 0.0,
            "V_flow_5": 0.0,
            "p_rel_5": 0.0,
            "V_flow_6": 0.0,
            "p_rel_6": 0.0,
        }
        self.outputs = {
            "w_p_1": 0.0,
            "w_v_2": 0.0,
            "w_v_3": 0.0,
            "w_p_4": 0.0,
            "w_v_5": 0.0,
            "w_v_6": 0.0,
        }  # rotational speed at the pump
        self.requested_volume_flow = 0.0  # setpoint for volume flow at valve
        self.error_flow = 0.0

    def do_step(self, time: float):  # mandatory method
        """This code is executed during each simulation step.

        Args:
            time (float): Simulated timestep

        """
        if time < 10:
            self.outputs["w_p_1"] = 0.9
        elif time < 20:
            self.outputs["w_p_1"] = 0.005
        elif time > 20:
            self.outputs["w_p_1"] = 0.9

        self.outputs["w_v_2"] = 0.5
        self.outputs["w_v_3"] = 0.5
        self.outputs["w_v_5"] = 0.5
        self.outputs["w_v_6"] = 0.5

        if time < 50:
            self.outputs["w_p_4"] = 0.9
        elif time < 60:
            self.outputs["w_p_4"] = 0.005
        elif time > 70:
            self.outputs["w_p_4"] = 0.9

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
    stop_time=100.0,
    step_size=1.0,
    fmu_paths=fmu_paths,
    model_classes=model_classes,
    connections_config=connections_config,
    parameters_to_log=parameters_to_log,
    logging_step_size=1,
    get_units=True,
)

# %% display results - consumer 6
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.plot(
    results["time"],
    results["water_network.V_flow_1"],
    lw=1.5,
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground=[77 / 255, 73 / 255, 67 / 255]),
        pe.Normal(),
    ],
    c=[253 / 255, 202 / 255, 0 / 255],
)
ax2.plot(
    results["time"],
    results["control_api.w_p_1"],
    lw=1.5,
    markersize=8,
    c=[0 / 255, 78 / 255, 115 / 255],
)
ax.set_xlabel("TIME in s")
ax.set_ylabel("VOLUME FLOW in m$^3$/h")
ax2.set_ylabel("PUMP ROTATIONAL SPEED in %", c=[0 / 255, 78 / 255, 115 / 255])
ax2.spines["right"].set_visible(True)

# %%
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
    results["control_api.w_p_4"],
    lw=1.5,
    markersize=8,
    c=[0 / 255, 78 / 255, 115 / 255],
)
ax.set_xlabel("TIME in s")
ax.set_ylabel("VOLUME FLOW in m$^3$/h")
ax2.set_ylabel("PUMP ROTATIONAL SPEED in %", c=[0 / 255, 78 / 255, 115 / 255])
ax2.spines["right"].set_visible(True)
