from typing import List, Union

import numpy as np

from Sofirpy.networks.agents import AgentConfig, PumpAgent, ConsumerAgent


class SharedSpace:
    """Multi-Agent system class which is needed to handle every agents' communication and saving data."""

    def __init__(self):
        """Init empty MAS.
        """
        self.number_agents = 0
        self.all_agents: List[Union["PumpAgent", "ConsumerAgent"]] = []
        self.pump_agents: List["PumpAgent"] = []
        self.consumer_agents: List["ConsumerAgent"] = []
        self.time_index = 0
        self.control_step_interval = 10

    def _add_agent(self, agent_config: "AgentConfig"):
        """Adds agents to SharedSpace, based on the type of the agent.

        Args:
            agent_config (AgentConfig):  Data class with all the information required to instantiate the agents.
        """
        if agent_config.agent_type == "consumer":
            consumer_agent = ConsumerAgent(agent_config=agent_config)
            self.consumer_agents.append(consumer_agent)
            self.all_agents.append(consumer_agent)
        elif agent_config.agent_type == "pump":
            pump_agent = PumpAgent(agent_config=agent_config)
            self.pump_agents.append(pump_agent)
            self.all_agents.append(pump_agent)

        self.number_agents += 1

    def add_agents_from_configs(self, agents_configs: dict):
        """Creates all agents defined in agents_config_json and adds them to the SharedSpace.

        Args:
            agents_config_json (dict): Dict based on a .json-file containing type and in-/output of the agents.
        """
        for json_index, agent_config in enumerate(agents_configs):
            self._add_agent(agent_config=agent_config)

    def step(self, time: float, action: np.ndarray):
        """Performs the recurring actions of the different agent types based on the perception of its environment.

        Accepts the chosen actions for either just the pump agents or all agents.
        The actions must be given as an array:
        [0:2] Pump speeds for the pump agents (0.0 to 1.0)
        [2:6] Demand volume flows for the consumer agents (m^3/h) (optional)

        Args:
            time (float): Time step of the co-simulation in seconds.
            action (np.ndarray): Array with the chosen actions for the pump agents.
        """

        # for t=0.0 FMU returns only none values, that is why it is not necessary to update
        # before the first iteration

        if time > 0:
            if self.time_index % self.control_step_interval == 0:
                for agent in self.all_agents:
                    agent.write_FMU_data()

                # demand volume flow is input to PI-controller
                for idx, consumer in enumerate(self.consumer_agents):
                    consumer.set_action(consumer.demand_volume_flow_m3h)

                for idx, pump in enumerate(self.pump_agents):
                    chosen_speed = float(action[idx])
                    pump.set_action(chosen_speed)
                self.time_index = 0

            else:
                for agent in self.all_agents:
                    agent.set_action(agent.old_action)

                self.time_index += 1




