import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from Mushroom.environments.fluid.minimal_tank_network import MinimalTankNetwork
from Sofirpy.networks.tank_minimal.controller import ControllerMinimalTank
from Sofirpy.simulation import ManualStepSimulator

stop_time = 55_000.0


class ControllerMinimalTankBugTest(ControllerMinimalTank):
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
            self.outputs["w_v_5"] = 0.3


def config():
    fmu_dir_path = Path(__file__).parent.parent.parent / "fluid_models" / "mini_water_network_tank"
    fmu_path = fmu_dir_path / "mini_tank.fmu"

    connections_config_path = fmu_dir_path / "mini_tank_connections_config.json"
    with open(connections_config_path) as connections_config_json:
        connections_config = json.load(connections_config_json)

    logging_config_path = fmu_dir_path / "mini_tank_parameters_to_log.json"
    with open(logging_config_path) as logging_config_json:
        parameters_to_log = json.load(logging_config_json)

    model_classes = {"control_api": ControllerMinimalTankBugTest}
    fmu_paths = {"water_network": str(fmu_path)}

    return {
        "fmu_paths": fmu_paths,
        "model_classes": model_classes,
        "model_init_args": {},
        "connections_config": connections_config,
        "parameters_to_log": parameters_to_log,
    }


sim = ManualStepSimulator(
    stop_time=stop_time,
    step_size=1.0,
    **config(),
    start_values={
        "water_network": {
            "tank_9.crossArea": 1.25,
            "tank_9.height": 2,
            "init_level_tank_9": 1,
            "elevation_tank_9": 7,
        }
    },
    logging_step_size=1,
    verbose=False,
    ignore_warnings=False,
)
sim.reset_simulation(
    stop_time=stop_time,
    step_size=1.0,
)
for time in tqdm(range(int(stop_time))):
    if time < 8_000:
        action = np.array([1.0, 1.0])
    elif time < 45_000:

        # TODO Diesen Wert ändern für die verschiedenen Tests:
        # 1.0 - Der Tank füllt sich nicht, weil er vorher leer wird
        # 0.9 - Der Tank füllt sich nicht, weil er vorher leer wird, hier wird zudem ein error geworfen
        # 0.85 - Der Tank füllt sich, er wurde vorher aber auch nicht ganz leer
        a = 0.85

        action = np.array([1.0, a])
    elif time > 45_000:
        action = np.array([1.0, 1.0])
    try:
        sim.do_simulation_step(action)
    except Exception as e:
        print("-------- Caught exception --------")
        print(e)
        break

MinimalTankNetwork._render_task(sim.get_results())
sim.finalize()
