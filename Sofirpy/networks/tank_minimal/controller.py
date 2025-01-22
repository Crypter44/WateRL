from sofirpy import SimulationEntity


class ControllerMinimalTank(SimulationEntity):
    """This Class is used when generating the input values for the FMU.

    It connects the input and output values for the FMU to a custom code.
    """

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

    def do_step(self, time: float):  # mandatory method
        """This code is executed during each simulation step.

        Args:
            time (float): Simulated timestep

        """
        if time < 10000:
            # Bedarf ist so gering, dass Tank befüllt werden kann
            self.outputs["w_p_4"] = 1
            self.outputs["w_v_5"] = 0.2
            self.outputs["w_v_7"] = 1
        elif time < 20000:
            # Bedarf ist so groß, dass Tank als zusätzliche Quelle eingesetzt wird
            self.outputs["w_p_4"] = 1
            self.outputs["w_v_5"] = 3
            self.outputs["w_v_7"] = -1
        elif time > 20000:
            self.outputs["w_p_4"] = 1
            self.outputs["w_v_5"] = 2
            self.outputs["w_v_7"] = 0

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
        print("Concluded simulation!")
