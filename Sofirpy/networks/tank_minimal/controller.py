
import numpy as np

from Sofirpy.simulation import SimulationEntityWithAction


class ControllerMinimalTank(SimulationEntityWithAction):
    """This Class is used when generating the input values for the FMU.

    It connects the input and output values for the FMU to a custom code.
    """
    CONTROL_STEP_INTERVAL = 1000

    def __init__(self) -> None:
        self.inputs = {
            "V_flow_4": 0.0,
            "p_rel_4": 0.0,
            "P_pum_4": 0.0,
            "V_flow_5": 0.0,
            "p_rel_5": 0.0,
            "u_v_5": 0.0,
            "V_flow_7_1": 0.0,
            "V_flow_7_2": 0.0,
            "p_rel_7_1": 0.0,
            "p_rel_7_2": 0.0,
            "u_v_7:1": 0.0,
            "u_v_7_2": 0.0,
            "level_tank_9": 0.0,
        }
        self.outputs = {
            "w_p_4": 0.0,
            "w_v_5": 0.0,
            "w_v_7": 0.0,
        }

    def do_step_with_action(self, time: float, action: np.ndarray):
        """This code is executed during each simulation step.

        Args:
            time (float): Simulated timestep

        """
        if time % self.CONTROL_STEP_INTERVAL == 0:
            self.outputs["w_p_4"] = float(action[0])  # agent 0 controls the pump's speed
            self.outputs["w_v_7"] = float(action[1])  # agent 1 controls the tank's valve

        if time < 8_000:
            self.outputs["w_v_5"] = 0.3
        elif time < 45_000:
            self.outputs["w_v_5"] = 2.83
        elif time < 55_000:
            self.outputs["w_v_5"] = 2.95



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
        [8] outflow from the tank
        [9] tank control
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
            + [self.get_parameter_value("V_flow_7_1")]
            # outflow from the tank
            + [self.get_parameter_value("V_flow_7_2")]
            # tank control
            + [self.outputs["w_v_7"]]
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
