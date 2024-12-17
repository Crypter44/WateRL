# %% import
import matplotlib.pyplot as plt
from pathlib import Path
import time

from sofirpy import simulate, SimulationEntity

dir_path = Path(__file__).parent
# plt.style.use(dir_path / "FST.mplstyle")

# %% setup simulation

fmu_dir_path = dir_path.parent / "Fluid_Model" / "minimal_example"
fmu_path = fmu_dir_path / "TestCase.fmu"

connections_config = {
    "water_network": [
        {
            "parameter_name": "u1",
            "connect_to_system": "control_api",
            "connect_to_external_parameter": "u1",
        }
    ],
    "control_api": [
        {
            "parameter_name": "y1",
            "connect_to_system": "water_network",
            "connect_to_external_parameter": "y1",
        }
    ],
}

parameters_to_log = {
    "water_network": [
        "y1",
    ],
    "control_api": ["u1"],
}


# %%
class Controler(SimulationEntity):
    """This Class is used when generating the input values for the FMU.

    It connects the input and output values for the FMU to a custom code.

    Args:
        SimulationEntity: Abstract object representing a simuation entity
    """

    def __init__(self) -> None:
        self.inputs = {
            "y1": 0.0,
        }
        self.outputs = {
            "u1": 0,
        }  # rotational speed at the pump

    def do_step(self, time: float):  # mandatory method
        """This code is executed during each simulation step.

        Args:
            time (float): Simulated timestep

        """
        if time < 10:
            self.outputs["u1"] = 0.9
        elif time < 20:
            self.outputs["u1"] = 0.005
        elif time > 20:
            self.outputs["u1"] = 0.9

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
    stop_time=30.0,
    step_size=1.0,
    fmu_paths=fmu_paths,
    model_classes=model_classes,
    connections_config=connections_config,
    parameters_to_log=parameters_to_log,
    logging_step_size=1,
    get_units=True,
)

# %%
fig, ax = plt.subplots()
ax.scatter(
    results["time"],
    results["water_network.y1"],
    lw=1.5,
)
