import json
from pathlib import Path

import numpy as np
from mushroom_rl.utils import spaces

from Mushroom.fluid_network_environments.fluid_network_environment import AbstractFluidNetworkEnv
from Sofirpy.networks.control_api import ControlApiCircular
from Sofirpy.simulation import ManualStepSimulator

working_dir = Path(__file__).parent.parent.parent

fmu_dir_path = working_dir / "Fluid_Model" / "circular_water_network"

fmu_path = fmu_dir_path / "mini_circular_water_network.fmu"

agent_config_path = fmu_dir_path / "mini_circular_water_network.json"

connections_config_path = fmu_dir_path / "mini_circular_water_network_connections_config.json"

logging_config_path = fmu_dir_path / "mini_circular_water_network_parameters_to_log.json"

# create interface of multi-agent system to FMU
model_classes = {"control_api": ControlApiCircular}
fmu_paths = {"water_network": str(fmu_path)}

with open(connections_config_path) as connections_config_json:
    connections_config = json.load(connections_config_json)

with open(logging_config_path) as logging_config_json:
    parameters_to_log = json.load(logging_config_json)


class CircularFluidNetwork(AbstractFluidNetworkEnv):
    def __init__(self, gamma: float):
        super().__init__(
            # 4 demands of the network, 4 resulting volume flows
            observation_space=spaces.Box(low=-10, high=10, shape=(8,)),
            # 2 rotaional speeds of the pumps
            action_space=spaces.Box(low=0, high=1, shape=(2,)),
            fluid_network_simulator=ManualStepSimulator(
                stop_time=200,
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
            horizon=200,
        )

    def _get_current_state(self):
        global_state, done = self.sim.get_current_state()
        try:
            controller_state = global_state["control_api"]
            return controller_state, done
        except KeyError:
            raise KeyError("The key 'control_api' was not found in the global state.")

    def _reward_fun(self, state: np.ndarray, action: np.ndarray, new_state: np.ndarray):
        pass  # TODO: implement reward function
