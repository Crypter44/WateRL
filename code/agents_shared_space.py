import numpy as np
import random

from agents import AgentConfig, PumpAgent, ConsumerAgent


class SharedSpace:
    """Multi-Agent system class which is needed to handle every agents' communication and saving data."""

    def __init__(self, project_directory: str):
        """Init empty MAS.

        Args:
            project_directory (str): Current project directory
        """
        self.number_agents = 0
        self.all_agents = []
        self.pump_agents = []
        self.consumer_agents = []
        self.project_directory = project_directory
        self.time_index = 0

    def _add_agent(self, agent_config: "AgentConfig"):
        """Adds agents to SharedSpace, based on the type of the agent.

        Args:
            agent_config (AgentConfig):  Data class with all the information required to instantiate the agents.
        """
        if agent_config.agent_type == "consumer":
            agent = ConsumerAgent(agent_config=agent_config)
            self.consumer_agents.append(agent)
        elif agent_config.agent_type == "pump":
            agent = PumpAgent(agent_config=agent_config)
            self.pump_agents.append(agent)

        self.all_agents.append(agent)
        self.number_agents += 1

    def add_agents_from_file(self, agents_config_json: dict):
        """Creates all agents defined in agents_config_json and adds them to the SharedSpace.

        Args:
            agents_config_json (dict): Dict based on a .json-file containing type and in-/output of the agents.
        """
        for json_index, agent_name in enumerate(agents_config_json["agents"].keys()):
            agent_config_json = agents_config_json["agents"][agent_name]

            agent_config = AgentConfig(
                name=agent_name,
                agent_id=json_index,
                agent_type=agent_config_json["type"],
                names_output_to_FMU=agent_config_json["input_FMU"],
                names_input_from_FMU=agent_config_json["output_FMU"],
            )

            self._add_agent(agent_config=agent_config)

    def step(self, time: float):
        """Performs the recurring actions of the different agent types based on the perception of its environment.

        Args:
            time (float): Time step of the co-simulation in seconds.
        """

        # for t=0.0 FMU returns only none values, that is why it is not necessary to update
        # before the first iteration

        if time > 0:
            if self.time_index == 9:
                for agent in self.all_agents:
                    agent.write_FMU_data(time)

                # demand volume flow is input to PI-controller
                for consumer in self.consumer_agents:
                    consumer.calculate_demand_volume_flow()
                    consumer.set_action(consumer.demand_volume_flow_m3h)

                for pump in self.pump_agents:
                    chosen_speed = random.choice(np.linspace(0, 1, 10))
                    pump.set_action(chosen_speed)
                self.time_index = 0

            elif self.time_index != 9:
                for agent in self.all_agents:
                    agent.set_action(agent.old_action)

                self.time_index += 1
