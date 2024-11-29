import logging
import os
import shutil
import tempfile
from abc import abstractmethod
from typing import final

import numpy as np
import sofirpy.common as co
from sofirpy import SimulationEntity
from sofirpy.simulation.simulation import Simulator, _validate_input, init_fmus, \
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

    @abstractmethod
    def get_state(self):
        """Return the current state information, which this entity is aware of."""

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
            stop_time: float,
            step_size: float,
            fmu_paths: co.FmuPaths | None = None,
            model_classes: co.ModelClasses | None = None,
            connections_config: co.ConnectionsConfig | None = None,
            start_values: co.StartValues | None = None,
            parameters_to_log: co.ParametersToLog | None = None,
            logging_step_size: float | None = None,
            get_units: bool = False,
            verbose: bool = False,
    ):
        """Initialize the simulator."""

        # Set the temporary directory for the FMUs, since fmpy creates a data leak otherwise
        self.tmp_dir = os.path.abspath("./fmu_tmp")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        logging.info("Cleared temporary directory.")
        tempfile.tempdir = self.tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)

        # Set the logging level to INFO if verbose is True, to get more information about the simulation
        if verbose:
            logging.basicConfig(
                format="Simulation::%(levelname)s::%(message)s",
                level=logging.INFO,
                force=True,
            )

        _validate_input(stop_time, step_size, fmu_paths, model_classes, connections_config, parameters_to_log,
                        logging_step_size, start_values, )

        # Save the parameters for a possible reset
        self._fmu_paths = fmu_paths or {}
        self._model_classes = model_classes or {}
        self._connections_config = connections_config or {}
        self._start_values = start_values or {}
        self._given_parameters_to_log = parameters_to_log or {}

        # Initialize attributes
        self.systems = {}
        self.fmus = None
        self.models = None
        self._get_units = get_units

        self._time_series = None
        self._time_step = None

        self._number_log_steps = None
        self._logging_multiple = None
        self._log_step = None
        self._parameters_to_log = None

        self.results = None

        self.reset_simulation(stop_time, step_size, logging_step_size)
        super().__init__(self.systems, self.connections, self._parameters_to_log)

    def reset_simulation(
            self,
            stop_time: float,
            step_size: float,
            logging_step_size: float | None = None,
            start_time: float = 0.0,
    ) -> None:
        """Reset the simulation to the initial state."""
        self.conclude_simulation()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        logging.info("Cleared temporary directory.")
        os.makedirs(self.tmp_dir, exist_ok=True)

        stop_time = float(stop_time)
        step_size = float(step_size)

        logging.info(f"Simulation stop time set to {stop_time} seconds.")
        logging.info(f"Simulation step size set to {step_size} seconds.")

        logging_step_size = float(logging_step_size or step_size)

        logging.info(f"Simulation logging step size set to {logging_step_size} seconds.")

        self.fmus = init_fmus(self._fmu_paths, step_size, self._start_values.copy())
        self.models = init_models(self._model_classes, self._start_values.copy())

        # Check if all user-defined python models are SimulationEntityWithAction
        for model in self.models.values():
            if not isinstance(model.simulation_entity, SimulationEntityWithAction):
                raise TypeError(f"Model '{model}' is not a SimulationEntityWithAction.")

        self.systems = {**self.fmus, **self.models}
        self.connections = init_connections(self._connections_config)
        self._parameters_to_log = init_parameter_list(self._given_parameters_to_log)
        self.parameters_to_log = self._parameters_to_log

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
        if self.is_done():
            return

        time_step, time = self._time_step, self._time_series[self._time_step]
        for fmu in self.fmus.values():
            fmu.simulation_entity.do_step(time)
        for agent in self.models.values():
            agent.simulation_entity.do_step_with_action(time, action)
        self.set_systems_inputs()
        if ((time_step + 1) % self._logging_multiple) == 0:
            self.log_values(self._time_series[time_step + 1], self._log_step)
            self._log_step += 1
        self._time_step += 1

    def get_current_state(self):
        """Return the current state of the simulation."""
        state = {}
        for agent in self.models.values():
            state[agent.name] = agent.simulation_entity.get_state()

        return state, self.is_done()

    def is_done(self) -> bool:
        """Return True iff the simulation is finished."""
        return self._time_step == len(self._time_series) - 1

    def get_results(self):
        """Return the simulation results."""
        if self._get_units:
            return self.convert_to_data_frame(self.results), self.get_units()
        return self.convert_to_data_frame(self.results)

    def finalize(self):
        """Finalize the simulation"""
        self.conclude_simulation()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        logging.info("Cleared temporary directory.")
        logging.info("Simulation finished.")
