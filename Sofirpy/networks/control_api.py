import json
import logging
from pathlib import Path

from Sofirpy.networks.agents_shared_space import SharedSpace, SharedSpaceWithoutPi
from Sofirpy.simulation import SimulationEntityWithAction


class ControlApiCircular(SimulationEntityWithAction):
    """This Class is used when generating the input values for the FMU.

    It connects the input and output values for the FMU to a custom code.
    """

    def __init__(self):
        """_summary_"""
        working_dir = Path(__file__).parent.parent.parent
        self.project_dir = working_dir / "Fluid_Model"
        # TODO: get the data from script calling the class
        agent_config_path = (
                self.project_dir
                / "circular_water_network"
                / "mini_circular_water_network.json"
        )
        logging.info("agent information taken from {}".format(agent_config_path))

        # read agent configuration from json-file
        with open(agent_config_path) as agent_config_json:
            agents_config_data = json.load(agent_config_json)
            agent_config_json.close()

        # create multi-agent system
        self.mas = SharedSpace(project_directory=self.project_dir)
        self.mas.add_agents_from_file(agents_config_data)

    def do_step_with_action(self, time: float, action):
        self.mas.step(time, action)

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
                + [self.get_parameter_value(f"u_v_{i}") for i in [2, 3, 5, 6]]
                # rotational speeds of the pumps
                + [p.speed_to_FMU for p in self.mas.pump_agents]
                # volume flows at the pumps
                + [self.get_parameter_value(f"V_flow_{i}") for i in [1, 4]]
                # power consumption of the pumps
                + [self.get_parameter_value(f"P_pum_{i}") for i in [1, 4]]
        )
        return state

    def set_parameter(
            self, parameter_name: str, parameter_value: float
    ):  # mandatory method
        """Gets parameters from the FMU.

        Args:
            parameter_name (str): Name of the value as given in the connections_config.
            parameter_value (float): Value of the parameter.
        """
        for agent in self.mas.all_agents:
            for key in agent.inputs_from_FMU.keys():
                if parameter_name == key:
                    agent.inputs_from_FMU[key] = parameter_value

    def get_parameter_value(self, output_name: str) -> float:  # mandatory method
        """Extracts parameters that are imposed on the FMU.

        Args:
            output_name (str): Name of the value as given in the connections_config.

        Returns:
            float: Value of the parameter.
        """
        output_value = None
        for agent in self.mas.all_agents:
            for key in agent.output_to_FMU.keys():
                if output_name == key:
                    output_value = agent.output_to_FMU[key]
            for key in agent.inputs_from_FMU.keys():
                if output_name == key:
                    output_value = agent.inputs_from_FMU[key]
        if output_value is None:
            print(("Output variable '{}' has no value").format(output_name))
            output_value = 0
        return output_value

    def conclude_simulation(self):  # optional
        """Just to make sure."""
        # print("Concluded simulation!")
        pass


class ControlApiCircularNoPI(ControlApiCircular):
    def __init__(self):
        """_summary_"""
        working_dir = Path(__file__).parent.parent.parent
        self.project_dir = working_dir / "Fluid_Model"
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
        self.mas.add_agents_from_file(agents_config_data)

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
