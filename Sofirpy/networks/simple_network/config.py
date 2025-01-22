from pathlib import Path

from Sofirpy.networks.simple_network.controllersimplenetworkvalve import ControllerSimpleNetworkValve


def get_simple_network_valve_config():
    connections_config = {
        "water_network": [
            {
                "parameter_name": "w_v_2",
                "connect_to_system": "control_api",
                "connect_to_external_parameter": "w_v_2",
            }
        ],
        "control_api": [
            {
                "parameter_name": "P_pum_1",
                "connect_to_system": "water_network",
                "connect_to_external_parameter": "P_pum_1",
            },
            {
                "parameter_name": "V_flow_1",
                "connect_to_system": "water_network",
                "connect_to_external_parameter": "V_flow_1",
            },
            {
                "parameter_name": "p_rel_1",
                "connect_to_system": "water_network",
                "connect_to_external_parameter": "p_rel_1",
            },
            {
                "parameter_name": "V_flow_2",
                "connect_to_system": "water_network",
                "connect_to_external_parameter": "V_flow_2",
            },
            {
                "parameter_name": "p_rel_2",
                "connect_to_system": "water_network",
                "connect_to_external_parameter": "p_rel_2",
            },
        ],
    }

    parameters_to_log = {
        "water_network": [
            "V_flow_1",
            "p_rel_1",
            "P_pum_1",
            "V_flow_2",
            "p_rel_2",
        ],
        "control_api": ["w_v_2"],
    }

    dir_path = Path(__file__).parent
    fmu_dir_path = dir_path.parent.parent / "fluid_models" / "simple_network_valve"
    fmu_path = fmu_dir_path / "simple_network_valve.fmu"

    model_classes = {"control_api": ControllerSimpleNetworkValve}
    fmu_paths = {"water_network": str(fmu_path)}

    return fmu_paths, model_classes, connections_config, parameters_to_log
