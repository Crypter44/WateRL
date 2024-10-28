# %% import
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from Sofirpy.step_by_step_simulation import SimulationEntityWithAction, setup_manual_step_simulation

dir_path = Path(__file__).parent

# %% setup simulation

fmu_dir_path = dir_path.parent / "Fluid_Model" / "simple_network_valve"
fmu_path = fmu_dir_path / "simple_network_valve.fmu"

connections_config = {
    "water_network": [
        {
            "parameter_name": "w_v_2",
            "connect_to_system": "control_api",
            "connect_to_external_parameter": "w_v_2",
        }
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
    ],
}

parameters_to_log = {
    "water_network": [
        "V_flow_1",
        "p_rel_1",
        "P_pum_1",
        "V_flow_2",
        "p_rel_2",
    ],
    "control_api": ["w_v_2"],
}


# %%
class Controler(SimulationEntityWithAction):
    """This Class is used when generating the input values for the FMU.

    It connects the input and output values for the FMU to a custom code.

    Args:
        SimulationEntity: Abstract object representing a simuation entity
    """

    def __init__(self) -> None:
        self.inputs = {
            "V_flow_1": 0.0,
            "p_rel_1": 0.0,
            "P_pum_1": 0.0,
            "V_flow_2": 0.0,
            "p_rel_2": 0.0,
        }
        self.outputs = {"w_v_2": 0.0}  # rotational speed at the pump
        self.requested_volume_flow = 0.0  # setpoint for volume flow at valve
        self.error_flow = 0.0

    def do_step_with_action(self, time: float, action: np.ndarray):  # mandatory method
        """This code is executed during each simulation step.

        Args:
            time (float): Simulated timestep

        """
        self.outputs["w_v_2"] = float(action[0])
        return self.inputs["V_flow_2"]


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

# create simulation
sim = setup_manual_step_simulation(
    stop_time=100,
    step_size=1,
    fmu_paths=fmu_paths,
    model_classes=model_classes,
    connections_config=connections_config,
    parameters_to_log=parameters_to_log,
)

# Run the simulation
out = 0
while not sim.is_done():
    # calculate the control action
    requested_volume_flow = 0.5
    error_flow = requested_volume_flow - sim.systems["water_network"].simulation_entity.get_parameter_value("V_flow_2")
    u = out + 0.1 * error_flow

    # limitation of actions
    if u > 1:
        u = 1
    if u < 0:
        u = 0

    out = u

    # do simulation step with the calculated action
    actual_vf = sim.do_simulation_step(np.array([u]))

# finalize simulation and get results
results = sim.finalize()

# %% display results - consumer 6
fig, ax = plt.subplots(figsize=(10, 8))
ax2 = ax.twinx()

ax.plot(
    np.linspace(0, 100, 100),
    [0.5] * 100,
    lw=1.5,
    label="DEMAND",
    linestyle=(0, (2, 1)),
    c = [0 / 255, 78 / 255, 115 / 255],
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
