import json
from pathlib import Path

from Sofirpy.networks.agents import AgentConfig
from Sofirpy.networks.circular_network.controller import ControllerCircular


def get_circular_network_config():
    fmu_dir_path = Path(__file__).parent.parent.parent / "fluid_models" / "circular_water_network"
    fmu_path = fmu_dir_path / "mini_circular_water_network.fmu"
    connections_config_path = fmu_dir_path / "mini_circular_water_network_connections_config.json"
    logging_config_path = fmu_dir_path / "mini_circular_water_network_parameters_to_log.json"
    agent_configs_path = fmu_dir_path / "mini_circular_water_network.json"

    # create interface of multi-agent system to FMU
    model_classes = {"control_api": ControllerCircular}
    fmu_paths = {"water_network": str(fmu_path)}

    with open(connections_config_path) as connections_config_json:
        connections_config = json.load(connections_config_json)

    with open(logging_config_path) as logging_config_json:
        parameters_to_log = json.load(logging_config_json)

    with open(agent_configs_path) as agent_configs_json:
        agent_configs_dict = json.load(agent_configs_json)
        agent_configs = []
        for idx, agent_name in enumerate(agent_configs_dict["agents"].keys()):
            agent_configs.append(
                AgentConfig(
                    name=agent_name,
                    agent_id=idx,
                    agent_type=agent_configs_dict["agents"][agent_name]["type"],
                    names_output_to_FMU=agent_configs_dict["agents"][agent_name]["output_FMU"],
                    names_input_from_FMU=agent_configs_dict["agents"][agent_name]["input_FMU"],
                )
            )

    model_init_args = {"control_api": {"agent_configs": agent_configs}}

    return {
        "fmu_paths": fmu_paths,
        "model_classes": model_classes,
        "model_init_args": model_init_args,
        "connections_config": connections_config,
        "parameters_to_log": parameters_to_log,
    }
