from mushroom_rl.utils import spaces

from Mushroom.fluid_network_environments.fluid_network_environment import AbstractFluidNetworkEnv
from Sofirpy.run_simple_network_valve_with_manual_step import fmu_paths, model_classes, connections_config, \
    parameters_to_log
from Sofirpy.step_by_step_simulation import setup_manual_step_simulation


class SimpleNetworkValve(AbstractFluidNetworkEnv):
    def __init__(self, gamma: float, horizon: int, fluid_network_simulator_args: dict = None):
        super().__init__(
            observation_space=spaces.Box(low=-10, high=10, shape=(3,)),
            action_space=spaces.Box(low=0, high=1, shape=(1,)),
            fluid_network_simulator=setup_manual_step_simulation(
                stop_time=100,
                step_size=1,
                fmu_paths=fmu_paths,
                model_classes=model_classes,
                connections_config=connections_config,
                parameters_to_log=parameters_to_log,
            ),
            gamma=gamma,
            horizon=horizon,
            fluid_network_simulator_args=fluid_network_simulator_args,
        )

    def _get_current_state(self):
        global_state, done = self._sim.get_current_state()
        try:
            local_state = global_state["control_api"]
        except KeyError:
            raise KeyError("The key 'control_api' was not found in the global state.")
        return local_state, done

    def _reward_fun(self, state, action, new_state):
        demand = state[0]
        supply = state[1]

        new_demand = new_state[0]
        new_supply = new_state[1]

        return -abs(new_demand - new_supply)
