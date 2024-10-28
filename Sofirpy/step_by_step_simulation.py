import logging
from abc import abstractmethod
from typing import final

import numpy as np
import sofirpy.common as co
from sofirpy import SimulationEntity
from sofirpy.simulation.simulation import Simulator, System, Connection, SystemParameter, _validate_input, init_fmus, \
    init_models, init_connections, init_parameter_list


class SimulationEntityWithAction(SimulationEntity):
    """
    Interface for simulation entities which support actions.

    This interface extends the SimulationEntity interface by adding a method to perform a simulation step with a given
    action. The old do_step method is marked as final to prevent it from being called on entities which support actions.
    """
    @abstractmethod
    def do_step_with_action(self, time: float, action: np.ndarray):
        """Perform a simulation step with a given action."""

    @final
    def do_step(self, time: float) -> None:
        raise NotImplementedError("This method should not be called on an entity which supports actions.")


class ManualStepSimulator(Simulator):
    """
    Simulator which performs a simulation step by step, where the steps are user-controlled.

    This simulator is used to perform a simulation step by step,
    where the user has to call the do_simulation_step method to perform a single step.
    The old simulate method is marked as final to prevent it from being called on a manual step simulator.
    """
    def __init__(
            self,
            systems: dict[str, System],
            fmus: dict[str, System],
            agents: dict[str, System],
            connections: list[Connection],
            parameters_to_log: list[SystemParameter],
            return_units: bool = False
    ):
        super().__init__(systems, connections, parameters_to_log)
        self.fmus = fmus
        self.agents = agents
        self._time_series = None
        self._time_step = None
        self._number_log_steps = None
        self._logging_multiple = None
        self.results = None
        self._log_step = None
        self._return_units = return_units

    def reset_simulation(
            self,
            stop_time: float,
            step_size: float,
            logging_step_size: float,
            start_time: float = 0.0,
    ) -> None:
        """Reset the simulation to the initial state."""
        self._time_series = self.compute_time_array(stop_time, step_size, start_time)
        self._time_step = 0
        self._number_log_steps = int(stop_time / logging_step_size) + 1
        self._logging_multiple = round(logging_step_size / step_size)
        self.results = np.zeros((self._number_log_steps, len(self.parameters_to_log) + 1))
        self.log_values(time=0, log_step=0)
        self._log_step = 1
        logging.info("Reset simulation.")

    @final
    def simulate(
            self,
            stop_time: float,
            step_size: float,
            logging_step_size: float,
            start_time: float = 0.0,
    ):
        raise NotImplementedError("This method should not be called on a manual step simulator.")

    def do_simulation_step(
            self,
            action: np.ndarray,
    ):
        """Perform a single simulation step."""
        time_step, time = self._time_step, self._time_series[self._time_step]
        for fmu in self.fmus.values():
            fmu.simulation_entity.do_step(time)
        for agent in self.agents.values():
            agent.simulation_entity.do_step_with_action(time, action)
        self.set_systems_inputs()
        if ((time_step + 1) % self._logging_multiple) == 0:
            self.log_values(self._time_series[time_step + 1], self._log_step)
            self._log_step += 1
        self._time_step += 1

    def is_done(self) -> bool:
        """Return True iff the simulation is finished."""
        return self._time_step == len(self._time_series) - 1

    def finalize(self):
        """Finalize the simulation and return the results."""
        self.conclude_simulation()
        if self._return_units:
            return self.convert_to_data_frame(self.results), self.get_units()
        return self.convert_to_data_frame(self.results)


def setup_manual_step_simulation(
    stop_time: float,
    step_size: float,
    fmu_paths: co.FmuPaths | None = None,
    model_classes: co.ModelClasses | None = None,
    connections_config: co.ConnectionsConfig | None = None,
    start_values: co.StartValues | None = None,
    parameters_to_log: co.ParametersToLog | None = None,
    logging_step_size: float | None = None,
    get_units: bool = False,
):
    """Set up a simulation with manual step control."""
    logging.basicConfig(
        format="Simulation::%(levelname)s::%(message)s",
        level=logging.INFO,
        force=True,
    )
    _validate_input(
        stop_time,
        step_size,
        fmu_paths,
        model_classes,
        connections_config,
        parameters_to_log,
        logging_step_size,
        start_values,
    )

    stop_time = float(stop_time)
    step_size = float(step_size)

    logging.info(f"Simulation stop time set to {stop_time} seconds.")
    logging.info(f"Simulation step size set to {step_size} seconds.")

    logging_step_size = float(logging_step_size or step_size)

    logging.info(f"Simulation logging step size set to {logging_step_size} seconds.")

    connections_config = connections_config or {}
    fmu_paths = fmu_paths or {}
    model_classes = model_classes or {}
    start_values = start_values or {}
    parameters_to_log = parameters_to_log or {}

    start_values = start_values.copy()  # copy because dict will be modified in fmu.py

    fmus = init_fmus(fmu_paths, step_size, start_values)

    models = init_models(model_classes, start_values)

    # Check if all user-defined python models are SimulationEntityWithAction
    for model in models.values():
        if not isinstance(model.simulation_entity, SimulationEntityWithAction):
            raise TypeError(f"Model '{model}' is not a SimulationEntityWithAction.")

    connections = init_connections(connections_config)
    _parameters_to_log = init_parameter_list(parameters_to_log)

    # Create the simulator
    sim = ManualStepSimulator(
        {**fmus, **models},
        fmus,
        models,
        connections,
        _parameters_to_log,
        return_units=get_units
    )
    # Reset the simulation to the initial state
    sim.reset_simulation(stop_time, step_size, logging_step_size)
    return sim