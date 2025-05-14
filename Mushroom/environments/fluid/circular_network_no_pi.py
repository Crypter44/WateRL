import warnings

from mushroom_rl.utils import spaces

from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Sofirpy.networks.circular_network_no_pi.config import get_circular_network_no_pi_config
from Sofirpy.simulation import ManualStepSimulator
from mushroom_rl.utils import spaces

from Mushroom.environments.fluid.circular_network import CircularFluidNetwork
from Sofirpy.networks.circular_network_no_pi.config import get_circular_network_no_pi_config
from Sofirpy.simulation import ManualStepSimulator


class CircularFluidNetworkWithoutPI(CircularFluidNetwork):
    """
    WARNING: NOT SUPPORTED ANYMORE

    This environment is a circular fluid network without PI controllers.
    Instead, the valves are also controlled by the agents.
    """
    def __init__(
            self,
            observation_space=spaces.Box(low=-10, high=10, shape=(4,)),
            action_space=spaces.Box(low=0, high=1, shape=(1,)),
            fluid_network_simulator=ManualStepSimulator(
                stop_time=50,
                step_size=1,
                config=get_circular_network_no_pi_config(),
                logging_step_size=1,
                get_units=False,
                verbose=False,
            ),
            gamma: float = 0.99,
            criteria=None,
    ):
        warnings.warn("This environment is not supported, use at your own risk.")
        super().__init__(
            observation_spaces=[observation_space],
            action_spaces=[action_space],
            fluid_network_simulator=fluid_network_simulator,
            gamma=gamma,
            horizon=50,
            criteria=criteria,
        )

    def render(self, title=None, save_path=None):
        results = self.sim.get_results()
        valves = [2, 3, 5, 6]
        pumps = [1, 4]
        self.plot_valve_and_pump_data(
            results["time"],
            valves=valves,
            valve_openings=[results[f"control_api.w_v_{v}"] for v in valves],
            valve_demands=[results[f"control_api.demand_v_{v}"] for v in valves],
            valve_flows=[results[f"water_network.V_flow_{v}"] for v in valves],
            pumps=pumps,
            pump_speeds=[results[f"control_api.w_p_{p}"] for p in pumps],
            pump_powers=[results[f"water_network.V_flow_{p}"] for p in pumps],
            pump_flows=[results[f"water_network.V_flow_{p}"] for p in pumps],
            title=title,
            save_path=save_path,
        )

    def _get_simulation_state(self):
        """
        Return the observable state of the environment.
        """
        state, absorbing = super()._get_simulation_state()
        return state[:4].reshape((4, 1)), absorbing

    def local_observation_space(self, agent_index: int):
        return spaces.Box(low=-10, high=10, shape=(1,))

    def local_action_space(self, agent_index: int):
        return spaces.Box(low=0, high=1, shape=(1,))
