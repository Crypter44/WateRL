import numpy as np

from Sofirpy.simulation import SimulationEntityWithAction


class ControllerSimpleNetworkValve(SimulationEntityWithAction):
    """This Class is used when generating the input values for the FMU.

    It connects the input and output values for the FMU to a custom code.
    """

    def __init__(self) -> None:
        self.inputs = {
            "V_flow_1": 0.0,
            "p_rel_1": 0.0,
            "P_pum_1": 0.0,
            "V_flow_2": 0.0,
            "p_rel_2": 0.0,
        }
        self.outputs = {"w_v_2": 0.0}  # valve opening
        self.requested_volume_flow = np.random.uniform(0.1, 0.55)  # setpoint for volume flow at valve
        self.error_flow = 0.0

    def do_step_with_action(self, time: float, action: np.ndarray):  # mandatory method
        """This code is executed during each simulation step.

        Args:
            time (float): Simulated timestep
            action (np.ndarray): Action to be taken by the FMU

        """
        self.outputs["w_v_2"] = float(action[0])

    def get_state(self):
        return np.array([self.requested_volume_flow, self.inputs["V_flow_2"], self.outputs["w_v_2"]])

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
        return self.outputs[output_name]

    def conclude_simulation(self):  # optional
        """Just to make sure."""
        pass