from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mushroom_rl.utils import spaces

from Mushroom.fluid_network_environments.fluid_network_environment import AbstractFluidNetworkEnv
from Sofirpy.simulation import SimulationEntityWithAction, ManualStepSimulator


class Controller(SimulationEntityWithAction):
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
        self.outputs = {"w_v_2": 0.0}  # valve opening
        self.requested_volume_flow = np.random.uniform(0.1, 0.55)  # setpoint for volume flow at valve
        self.error_flow = 0.0

    def do_step_with_action(self, time: float, action: np.ndarray):  # mandatory method
        """This code is executed during each simulation step.

        Args:
            time (float): Simulated timestep

        """
        self.outputs["w_v_2"] = float(action[0])

    def get_state(self):
        return np.array([self.requested_volume_flow, self.inputs["V_flow_2"], self.outputs["w_v_2"]])

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
        pass


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

dir_path = Path(__file__).parent
fmu_dir_path = dir_path.parent.parent / "Fluid_Model" / "simple_network_valve"
fmu_path = fmu_dir_path / "simple_network_valve.fmu"

model_classes = {"control_api": Controller}
fmu_paths = {"water_network": str(fmu_path)}


class SimpleNetworkValve(AbstractFluidNetworkEnv):
    def __init__(self, gamma: float, horizon: int, fluid_network_simulator_args: dict = None):
        super().__init__(
            observation_space=spaces.Box(low=-10, high=10, shape=(2,)),
            action_space=spaces.Box(low=0, high=1, shape=(1,)),
            fluid_network_simulator=ManualStepSimulator(
                stop_time=horizon,
                step_size=1,
                fmu_paths=fmu_paths,
                model_classes=model_classes,
                connections_config=connections_config,
                parameters_to_log=parameters_to_log,
                logging_step_size=1,
                get_units=False,
                verbose=False,
            ),
            gamma=gamma,
            horizon=horizon,
            fluid_network_simulator_args=fluid_network_simulator_args,
        )

    def render(self, title: str = None):
        dataframe = self.sim.get_results()
        demand = self.sim.systems["control_api"].simulation_entity.requested_volume_flow,

        last_non_zero_index = dataframe["time"].to_numpy().nonzero()[0]

        if len(last_non_zero_index) == 0:
            print("No data to plot.")
            return
        else:
            last_non_zero_index = last_non_zero_index[-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax2 = ax.twinx()

        ax.plot(
            dataframe["time"][1:last_non_zero_index],
            [demand] * (last_non_zero_index-1),
            lw=1.5,
            label="DEMAND",
            linestyle=(0, (2, 1)),
            c=[253 / 255, 202 / 255, 0 / 255],
        )
        ax.plot(
            dataframe["time"][1:last_non_zero_index],
            dataframe["water_network.V_flow_2"][1:last_non_zero_index],
            lw=1.5,
            label="VOLUME FLOW",
            c=[253 / 255, 202 / 255, 0 / 255],
        )
        ax2.plot(
            dataframe["time"][1:last_non_zero_index],
            dataframe["control_api.w_v_2"][1:last_non_zero_index],
            lw=1.5,
            label="VALVE OPENING",
            markersize=8,
            c=[0 / 255, 78 / 255, 115 / 255],
        )
        ax.set_xlabel("TIME in s")
        ax.set_xlim(0, self._horizon)
        ax.set_ylim(0, 0.65)
        ax.set_ylabel("VOLUME FLOW AT VALVE in m$^3$/h")
        ax2.set_ylabel("VALVE OPENING in %", c=[0 / 255, 78 / 255, 115 / 255])
        ax2.set_ylim(0, 1)
        ax2.spines["right"].set_visible(True)
        fig.suptitle(title)
        fig.legend()
        fig.show()

    def step(self, action):
        # clip action to action space
        action = np.clip(action, self._mdp_info.action_space.low, self._mdp_info.action_space.high)
        self.sim.do_simulation_step(action)
        new_state, absorbing = self._get_current_state()

        reward = self._reward_fun(self._current_state, action, new_state)

        self._current_state = new_state
        return self._current_state, reward, absorbing, {}

    def _get_current_state(self):
        global_state, done = self.sim.get_current_state()
        try:
            local_state = global_state["control_api"]
        except KeyError:
            raise KeyError("The key 'control_api' was not found in the global state.")
        return local_state[0:2], done

    def _reward_fun(self, state, action, new_state):
        demand = state[0]
        supply = state[1]

        new_demand = new_state[0]
        new_supply = new_state[1]

        return -100 * (new_demand - new_supply) ** 2
