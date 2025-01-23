import json
from pathlib import Path

from Sofirpy.networks.tank_minimal.controller import ControllerMinimalTank


def get_minimal_tank_network_config():
    fmu_dir_path = Path(__file__).parent.parent.parent / "fluid_models" / "mini_water_network_tank"
    fmu_path = fmu_dir_path / "mini_tank.fmu"

    connections_config_path = fmu_dir_path / "mini_tank_connections_config.json"
    with open(connections_config_path) as connections_config_json:
        connections_config = json.load(connections_config_json)

    logging_config_path = fmu_dir_path / "mini_tank_parameters_to_log.json"
    with open(logging_config_path) as logging_config_json:
        parameters_to_log = json.load(logging_config_json)

    model_classes = {"control_api": ControllerMinimalTank}
    fmu_paths = {"water_network": str(fmu_path)}

    return {
        "fmu_paths": fmu_paths,
        "model_classes": model_classes,
        "model_init_args": {},
        "connections_config": connections_config,
        "parameters_to_log": parameters_to_log,
    }
