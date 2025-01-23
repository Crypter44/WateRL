import logging
import shutil
from pathlib import Path
from typing import cast, Any

from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from fmpy.simulation import apply_start_values, settable_in_instantiated, settable_in_initialization_mode
from sofirpy import utils
import sofirpy.common as co
from sofirpy.simulation.fmu import Fmu, SetterFunction, GetterFunction
from sofirpy.simulation.simulation import System


class ResettableFmu(Fmu):
    def initialize(self, start_values: dict[str, co.StartValue]) -> None:
        """Initialize the fmu.

        Args:
            start_time (float, optional): start time. Defaults to 0.
        """
        self.model_description = read_model_description(self.fmu_path)
        self.unzip_dir = Path(extract(self.fmu_path))
        self.reset(start_values)

    def reset(self, start_values: dict[str, co.StartValue]) -> None:
        """Initialize the fmu.

                Args:
                    start_time (float, optional): start time. Defaults to 0.
                """
        self.model_description_dict = {
            variable.name: variable
            for variable in self.model_description.modelVariables
        }

        self.fmu = FMU2Slave(
            guid=self.model_description.guid,
            unzipDirectory=self.unzip_dir,
            modelIdentifier=self.model_description.coSimulation.modelIdentifier,
            instanceName="instance1",
        )
        self.setter_functions: dict[str, SetterFunction] = {
            "Boolean": self.fmu.setBoolean,
            "Integer": self.fmu.setInteger,
            "Real": self.fmu.setReal,
            "Enumeration": self.fmu.setInteger,
        }
        self.getter_functions: dict[str, GetterFunction] = {
            "Boolean": self.fmu.getBoolean,
            "Integer": self.fmu.getInteger,
            "Real": self.fmu.getReal,
        }
        self.fmu.instantiate()
        self.fmu.setupExperiment()
        not_set_start_values = apply_start_values(
            self.fmu, self.model_description, start_values, settable_in_instantiated
        )
        self.fmu.enterInitializationMode()
        not_set_start_values = apply_start_values(
            self.fmu,
            self.model_description,
            not_set_start_values,
            settable_in_initialization_mode,
        )
        if not_set_start_values:
            logging.warning(
                f"The following start values for the FMU '{self.name}' "
                f"can not be set:\n{not_set_start_values}"
            )
        self.fmu.exitInitializationMode()

    def conclude_simulation(self) -> None:
        super().conclude_simulation()

    def finalize(self):
        shutil.rmtree(self.unzip_dir)
        logging.info(f"Cleared temp dir: '{self.unzip_dir}'.")


def init_fmus_resettable(
    fmu_paths: co.FmuPaths, step_size: float, start_values: co.StartValues
) -> dict[str, System]:
    """Initialize fmus as a System object and store them in a dictionary.

    Args:
        fmu_paths (FmuPaths): Dictionary which defines which fmu should be simulated.
            key -> name of the fmu; value -> path to the fmu
        step_size (float): step size of the simulation
        start_values (StartValues): Dictionary which defines start values for the
            systems.

    Returns:
        dict[str, System]: key -> fmu name; value -> System instance
    """
    fmus: dict[str, System] = {}
    for fmu_name, _fmu_path in fmu_paths.items():
        fmu_path: Path = utils.convert_str_to_path(_fmu_path, "fmu_path")
        fmu = ResettableFmu(fmu_path, fmu_name, step_size)
        _start_values = start_values.get(fmu_name) or {}
        fmu.initialize(start_values=_start_values)
        system = System(fmu, fmu_name)
        fmus[fmu_name] = system
        logging.info(f"FMU '{fmu_name}' initialized.")

    return fmus


def reset_fmus(
    fmus: dict[str, System], fmu_paths: co.FmuPaths, start_values: co.StartValues
) -> dict[str, System]:
    """Initialize fmus as a System object and store them in a dictionary.

    Args:
        fmus (dict[str, System]): Previously initialized fmus to be reset.
        fmu_paths (FmuPaths): Dictionary which defines which fmu should be simulated.
            key -> name of the fmu; value -> path to the fmu
        start_values (StartValues): Dictionary which defines start values for the
            systems.

    Returns:
        dict[str, System]: key -> fmu name; value -> System instance
    """
    for fmu_name, _fmu_path in fmu_paths.items():
        _start_values = start_values.get(fmu_name) or {}
        assert isinstance(fmus[fmu_name].simulation_entity, ResettableFmu), f"FMU '{fmu_name}' is not a ResettableFmu."
        resettable_fmu = cast(ResettableFmu, fmus[fmu_name].simulation_entity)
        resettable_fmu.reset(start_values=_start_values)
        logging.info(f"FMU '{fmu_name}' reset.")

    return fmus


def init_models(
    model_classes: co.ModelClasses,
    model_init_args: dict[str, dict[str, Any]],
    start_values: co.StartValues,
) -> dict[str, System]:
    """Initialize python models as a System object and store them in a dictionary.

    Args:
        model_classes (ModelClasses): Dictionary which defines which Python Models
            should be simulated.
        model_init_args (dict[str, list[Any]]): Dictionary which defines the arguments
            that should be passed to the Python Models.
        start_values (StartValues): Dictionary which defines start values for the
            systems.

    Returns:
        dict[str, System]: key -> python model name; value -> System instance
    """

    models: dict[str, System] = {}
    for model_name, model_class in model_classes.items():
        _start_values = start_values.get(model_name) or {}
        _init_args = model_init_args.get(model_name) or {}
        model_instance = model_class(**_init_args)
        model_instance.initialize(_start_values)
        system = System(model_instance, model_name)
        models[model_name] = system
        logging.info(f"Python Model '{model_name}' initialized.")

    return models