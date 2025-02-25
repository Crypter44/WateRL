
import numpy as np

from Sofirpy.simulation import SimulationEntityWithAction


class ControllerMinimalTank(SimulationEntityWithAction):
    """This Class is used when generating the input values for the FMU.

    It connects the input and output values for the FMU to a custom code.
    """
    CONTROL_STEP_INTERVAL = 300
    DEMAND_STEP_INTERVAL = 600
    ENSURE_POSSIBLE_DEMAND = True

    def __init__(self, demand_curve="tagesgang") -> None:
        self.inputs = {
            "V_flow_4": 0.0,
            "p_rel_4": 0.0,
            "P_pum_4": 0.0,
            "V_flow_5": 0.0,
            "p_rel_5": 0.0,
            "u_v_5": 0.0,
            "V_flow_7": 0.0,
            "p_rel_7": 0.0,
            "u_v_7": 0.0,
            "level_tank_9": 0.0,
        }

        if demand_curve == "tagesgang":
            omega = 2 * np.pi / 24
            time_demand_curve = np.linspace(0, 86_400, 86_401)
            scale = 1 / 3_600
            self.demand_curve = (
                    55  # daily demand in m3
                    # the following function gives a demand per time unit in % of the daily demand
                    # the total daily demand is reached after 24 time units
                    * (
                            4.21452027e-02
                            - 1.41579420e-02 * np.cos(omega * time_demand_curve * scale)
                            - 1.62752679e-02 * np.sin(omega * time_demand_curve * scale)
                            + 5.94972876e-03 * np.cos(2 * omega * time_demand_curve * scale)
                            - 1.82545802e-02 * np.sin(2 * omega * time_demand_curve * scale)
                            - 2.72810544e-03 * np.cos(3 * omega * time_demand_curve * scale)
                            + 2.15704832e-03 * np.sin(3 * omega * time_demand_curve * scale)
                            - 5.15308835e-03 * np.cos(4 * omega * time_demand_curve * scale)
                            - 1.10491878e-04 * np.sin(4 * omega * time_demand_curve * scale)
                    )
            )
        elif demand_curve == "tagesgang_noisy":
            omega = 2.65711342e-01
            time_demand_curve = np.linspace(0, 86_400, 86_401)
            scale = 1 / 3_600
            self.demand_curve = (
                    54.5  # daily demand in m3
                    # the following function gives a demand per time unit in % of the daily demand
                    # the total daily demand is reached after 24 time units
                    * (
                            4.21452027e-02
                            - 1.41579420e-02 * np.cos(omega * time_demand_curve * scale)
                            - 1.62752679e-02 * np.sin(omega * time_demand_curve * scale)
                            + 5.94972876e-03 * np.cos(2 * omega * time_demand_curve * scale)
                            - 1.82545802e-02 * np.sin(2 * omega * time_demand_curve * scale)
                            - 2.72810544e-03 * np.cos(3 * omega * time_demand_curve * scale)
                            + 2.15704832e-03 * np.sin(3 * omega * time_demand_curve * scale)
                            - 5.15308835e-03 * np.cos(4 * omega * time_demand_curve * scale)
                            - 1.10491878e-04 * np.sin(4 * omega * time_demand_curve * scale)
                    )
            )
            self.demand_curve += np.random.normal(0, 0.2, len(self.demand_curve))
        elif demand_curve == "tagesgang_24":
            omega = 2 * np.pi / 24
            time_demand_curve = np.linspace(0, 86_400, 86_401)
            scale = 1 / 3_600
            self.demand_curve = (
                    56  # daily demand in m3
                    # the following function gives a demand per time unit in % of the daily demand
                    # the total daily demand is reached after 24 time units
                    * (
                            0.04171752
                            - 0.01535547 * np.cos(omega * time_demand_curve * scale)
                            - 0.01584253 * np.sin(omega * time_demand_curve * scale)
                            + 0.0038021 * np.cos(2 * omega * time_demand_curve * scale)
                            - 0.01817066 * np.sin(2 * omega * time_demand_curve * scale)
                            - 0.00305664 * np.cos(3 * omega * time_demand_curve * scale)
                            + 0.00332155 * np.sin(3 * omega * time_demand_curve * scale)
                    )
            )
        elif demand_curve == "linear":
            self.demand_curve = np.linspace(0, 4.00, 86_401)
        elif demand_curve == "constant":
            self.demand_curve = np.ones(86_401) * 2.0
        else:
            raise ValueError("Unknown demand curve")

        self.outputs = {
            "w_p_4": 0.0,
            "w_v_5": self.demand_curve[0],
            "w_v_7": 0.0,
        }

    def do_step_with_action(self, time: float, action: np.ndarray):
        """This code is executed during each simulation step.

        Args:
            time (float): Simulated timestep

        """
        if time % self.CONTROL_STEP_INTERVAL == 0:
            self.outputs["w_p_4"] = float(1.3 * action[0])  # agent 0 controls the pump's speed
            self.outputs["w_v_7"] = float(action[1])  # agent 1 controls the tank's valve

        if self.inputs["level_tank_9"] < 0.025 and self.inputs["p_rel_7"] <= 1e-6:
            self.outputs["w_v_7"] = 0.0

        # demand of the valve
        if time % self.DEMAND_STEP_INTERVAL == 0:
            demand = self.demand_curve[int(time)]
            if self.ENSURE_POSSIBLE_DEMAND:
                demand = np.clip(demand, 0.0, 4.0)
            self.outputs["w_v_5"] = demand

    def get_state(self):
        """
        Returns the current state information:

        [0] demand of the valve
        [1] resulting volume flow at the valve
        [2] opening of the valve
        [3] rotational speed of the pump
        [4] volume flow at the pump
        [5] power consumption of the pump
        [6] level of the tank
        [7] inflow to the tank
        [8] tank control
        [9] pressure at the tank valve
        """
        state = (
            # demand of the valve
            [self.outputs["w_v_5"]]
            # resulting volume flow at the valve
            + [self.get_parameter_value("V_flow_5")]
            # opening of the valves
            + [self.get_parameter_value("u_v_5")]
            # rotational speeds of the pumps
            + [self.outputs["w_p_4"]]
            # volume flows at the pumps
            + [self.get_parameter_value("V_flow_4")]
            # power consumption of the pumps
            + [self.get_parameter_value("P_pum_4")]
            # level of the tank
            + [self.get_parameter_value("level_tank_9")]
            # inflow to the tank
            + [self.get_parameter_value("V_flow_7")]
            # tank control
            + [self.outputs["w_v_7"]]
            # pressure at the tank valve
            + [self.get_parameter_value("p_rel_7")]
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
        self.inputs[parameter_name] = parameter_value

    def get_parameter_value(self, output_name: str) -> float:  # mandatory method
        """Extracts parameters that are imposed on the FMU.

        Args:
            output_name (str): Name of the value as given in the connections_config.

        Returns:
            float: Value of the parameter.
        """
        if output_name in self.outputs:
            return self.outputs[output_name]
        elif output_name in self.inputs:
            return self.inputs[output_name]
        else:
            raise ValueError(f"Parameter {output_name} not found.")

    def conclude_simulation(self):  # optional
        """Just to make sure."""
        pass
