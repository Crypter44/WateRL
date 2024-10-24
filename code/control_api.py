import json
from pathlib import Path

from sofirpy import SimulationEntity

from agents_shared_space import SharedSpace


class ControlAPI(SimulationEntity):
    """_summary_

    Args:
        SimulationEntity (_type_): _description_
    """

    def __init__(self):
        """_summary_"""
        dir_path = Path(__file__).parent
        self.project_dir = dir_path.parent / "Fluid_Model"
        # TODO: get the data from script calling the class
        agent_config_path = (
            dir_path.parent / "Fluid_Model" / "mini_circular_water_network.json"
        )
        print("agent information taken from {}".format(agent_config_path))

        # read agent configuration from json-file
        with open(agent_config_path) as agent_config_json:
            agents_config_data = json.load(agent_config_json)
            agent_config_json.close()

        # create multi-agent system
        self.mas = SharedSpace(project_directory=self.project_dir)
        self.mas.add_agents_from_file(agents_config_data)

    # TODO
    # def initialize(self)

    def do_step(self, time: float):  # mandatory method
        """_summary_

        Args:
            time (float): _description_

        """
        self.mas.step(time)

    def set_parameter(
        self, parameter_name: str, parameter_value: float
    ):  # mandatory method
        """_summary_

        Args:
            parameter_name (str): _description_
            parameter_value (float): _description_
        """
        for agent in self.mas.all_agents:
            for key in agent.inputs_from_FMU.keys():
                if parameter_name == key:
                    agent.inputs_from_FMU[key] = parameter_value

    def get_parameter_value(self, output_name: str) -> float:  # mandatory method
        """_summary_

        Args:
            output_name (str): _description_

        Returns:
            float: _description_
        """
        output_value = None
        for agent in self.mas.all_agents:
            for key in agent.output_to_FMU.keys():
                if output_name == key:
                    output_value = agent.output_to_FMU[key]
        if output_value is None:
            print(("Output variable '{}' has no value").format(output_name))
        return output_value

    def conclude_simulation(self):  # optional
        """_summary_"""
        print("Concluded simulation!")
