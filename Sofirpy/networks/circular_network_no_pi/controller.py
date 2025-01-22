import json
import logging
from pathlib import Path

from Sofirpy.networks.circular_network.controller import ControllerCircular
from Sofirpy.networks.circular_network_no_pi.shared_space import SharedSpaceWithoutPi


class ControllerCircularNoPi(ControllerCircular):
    def __init__(self):
        """_summary_"""
        working_dir = Path(__file__).parent.parent.parent
        self.project_dir = working_dir / "fluid_models"
        # TODO: get the data from script calling the class
        agent_config_path = (
                self.project_dir
                / "circular_water_network_wo_PI"
                / "mini_circular_water_network_wo_PI.json"
        )
        logging.info("agent information taken from {}".format(agent_config_path))

        # read agent configuration from json-file
        with open(agent_config_path) as agent_config_json:
            agents_config_data = json.load(agent_config_json)
            agent_config_json.close()

        # create multi-agent system
        self.mas = SharedSpaceWithoutPi(project_directory=self.project_dir)
        self.mas.add_agents_from_configs(agents_config_data)

    def get_state(self):
        """
        Returns the current state information:

        [0:4] demands of the consumers
        [4:8] volume flows 2, 3, 5, 6 at the valves, 1 and 4 are at the pumps
        [8:12] opening of the valves
        [12:14] rotational speeds of the pumps
        [14:16] volume flows at the pumps
        [16:18] power consumption of the pumps
        """
        state = (
                # demands of the valves
                [c.demand_volume_flow_m3h for c in self.mas.consumer_agents]
                # resulting volume flows at the valves, 1 and 4 are at the pumps
                + [self.get_parameter_value(f"V_flow_{i}") for i in [2, 3, 5, 6]]
                # opening of the valves
                + [self.get_parameter_value(f"w_v_{i}") for i in [2, 3, 5, 6]]
                # rotational speeds of the pumps
                + [p.speed_to_FMU for p in self.mas.pump_agents]
                # volume flows at the pumps
                + [self.get_parameter_value(f"V_flow_{i}") for i in [1, 4]]
                # power consumption of the pumps
                + [self.get_parameter_value(f"P_pum_{i}") for i in [1, 4]]
        )
        return state