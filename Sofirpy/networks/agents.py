from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import random


@dataclass
class AgentConfig:
    name: str  # name of the agent.
    agent_id: int  # unique ID of agent
    agent_type: str  # type of agent
    names_output_to_FMU: list = field(
        default_factory=list
    )  # name of the output variable, which is used as an input to the FMU
    names_input_from_FMU: list = field(
        default_factory=list
    )  # name of the input variables


class Agent(ABC):
    def __init__(self, agent_config: "AgentConfig", agent_type: str):
        """Base class for all agents, regardless of their type (e.g. pump, valve)

        Args:
            agent_config (AgentConfig): Data class with all the information required to instantiate the agents.
        """
        # general settings for agents
        self.name = agent_config.name
        self.agent_id = agent_config.agent_id
        self.agent_type = agent_type
        self.old_action = 0.0

        self.output_to_FMU = {}
        for name in agent_config.names_output_to_FMU:
            self.output_to_FMU[name] = 0.0  # initial value
        self.inputs_from_FMU = {}
        for name in agent_config.names_input_from_FMU:
            self.inputs_from_FMU[name] = 0.0

    def set_action(self, action: float):
        """Set an action to the environment, given the value of the action.

        Args:
            action (float): Action to be taken, which is transfered to the FMU.
        """
        key = list(self.output_to_FMU.keys())[0]  # agent has only one output to the FMU
        self.output_to_FMU[key] = action
        self.old_action = action

        if self.agent_type == "consumer":
            self.volume_flow_m3h_to_PI = action
        elif self.agent_type == "pump":
            self.speed_to_FMU = action

    @abstractmethod
    def write_FMU_data(self, time):
        pass


class ConsumerAgent(Agent):
    """ValveAgent implements methods from abstract class Agent to provide functionality for Valves.

    Args:
        Agent (Agent): Base class for agents.
    """

    def __init__(self, agent_config: "AgentConfig"):
        """Inits ValveAgent with agent_config.

        Args:
            agent_config (AgentConfig):  Data class with all the information required to instantiate the agents.
        """
        super().__init__(agent_config, agent_type="consumer")
        self.demand_volume_flow_m3h = 0  # agent's demand in m^3/h
        self.demand_volume_flow_m3h = np.random.uniform(0.1, 0.8)  # agent's demand in m^3/h
        self.output_to_FMU[list(self.output_to_FMU.keys())[1]] = self.demand_volume_flow_m3h
        self.volume_flow_m3h_to_PI = 0  # Volume flow in m^3/h, which is transferred to the local PI as a setpoint
        self.measured_delta_pressure_bar = (
            0  # relative pressure difference over valve in bar from FMU
        )
        self.measured_volume_flow_m3h = 0  # volume flow through valve in m^3/h from FMU
        self.measured_valve_position = 0  # valve opening position in % from FMU

    def write_FMU_data(self):
        """Assigns the data from the inputs_from_FMU dict to specific attributes of the agent."""
        for key in self.inputs_from_FMU.keys():
            if "V_flow" in key:
                self.measured_volume_flow_m3h = self.inputs_from_FMU[key]
            elif "p_rel" in key:
                self.measured_delta_pressure_bar = self.inputs_from_FMU[key]
            elif "u_v" in key:
                self.measured_valve_position = self.inputs_from_FMU[key]

    def calculate_demand_volume_flow(self):
        """Calculates the current demand of the consumer based on a random value on the Tagesganglinie.

        The curve fallows data from  'DVGW Arbeitsblatt W 410 2008-12 Wasserbedarf - Kennwerte und Einflussgrößen'
        (Tagesganglinie eines städischen Versogungsgebiets - Figure C2). A random value from the curve is selected as
        current demand.
        """
        omega = 2.65711342e-01
        time_demand_curve = np.linspace(0, 24, 100)
        demand_volume_flow_curve = (
                20  # daily demand in m3
                # the following function gives a demand per time unit in % of the daily demand
                # the total daily demand is reached after 24 time units
                * (
                        4.21452027e-02
                        - 1.41579420e-02 * np.cos(omega * time_demand_curve)
                        - 1.62752679e-02 * np.sin(omega * time_demand_curve)
                        + 5.94972876e-03 * np.cos(2 * omega * time_demand_curve)
                        - 1.82545802e-02 * np.sin(2 * omega * time_demand_curve)
                        - 2.72810544e-03 * np.cos(3 * omega * time_demand_curve)
                        + 2.15704832e-03 * np.sin(3 * omega * time_demand_curve)
                        - 5.15308835e-03 * np.cos(4 * omega * time_demand_curve)
                        - 1.10491878e-04 * np.sin(4 * omega * time_demand_curve)
                )
        )
        self.demand_volume_flow_m3h = random.choice(demand_volume_flow_curve)


class PumpAgent(Agent):
    """PumpAgent implements methods from abstract class Agent to provide functionality for Pumps.

    Args:
        Agent (Agent): Base class for agents.
    """

    def __init__(self, agent_config: "AgentConfig"):
        """Inits PumpAgent with agent_config.

        Args:
            agent_config (AgentConfig):  Data class with all the information required to instantiate the agents.
        """
        super().__init__(agent_config, agent_type="pump")
        self.measured_delta_pressure_bar = (
            0  # relative pressure difference over the pump in bar from FMU
        )
        self.measured_volume_flow_m3h = 0  # volume flow through pump in m^3/h from FMU
        self.measured_power_consumption = 0
        self.speed_to_FMU = 0

    def write_FMU_data(self):
        """Assigns the data from the inputs_from_FMU dict to specific attributes of the agent."""
        for key in self.inputs_from_FMU.keys():
            if "V_flow" in key:
                self.measured_volume_flow_m3h = self.inputs_from_FMU[key]
            elif "p_rel" in key:
                self.measured_delta_pressure_bar = self.inputs_from_FMU[key]
            elif "P_pum" in key:
                self.measured_power_consumption = self.inputs_from_FMU[key]
