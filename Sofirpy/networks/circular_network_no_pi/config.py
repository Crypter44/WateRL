import json
from pathlib import Path

from Sofirpy.networks.circular_network_no_pi.controller import ControllerCircularNoPi


def get_circular_network_no_pi_config():
    fmu_dir_path = Path(__file__).parent.parent.parent / "fluid_models" / "circular_water_network_wo_PI"
    fmu_path = fmu_dir_path / "mini_circular_water_network_wo_PI.fmu"
    connections_config_path = fmu_dir_path / "mini_circular_water_network_wo_PI_connections_config.json"
    logging_config_path = fmu_dir_path / "mini_circular_water_network_wo_PI_parameters_to_log.json"

    # create interface of multi-agent system to FMU
    model_classes = {"control_api": ControllerCircularNoPi}
    fmu_paths = {"water_network": str(fmu_path)}

    with open(connections_config_path) as connections_config_json:
        connections_config = json.load(connections_config_json)

    with open(logging_config_path) as logging_config_json:
        parameters_to_log = json.load(logging_config_json)

    return fmu_paths, model_classes, connections_config, parameters_to_log
