import time

import numpy as np

from Sofirpy.networks.circular_network.config import get_circular_network_config
from Sofirpy.simulation import ManualStepSimulator


if __name__ == "__main__":
    num_tests = 1000
    num_steps_per_test = 100

    times = np.zeros((num_tests, 2))
    global_start = time.time()
    sim = ManualStepSimulator(
            stop_time=num_steps_per_test,
            step_size=1,
            **get_circular_network_config(),
            logging_step_size=1,
            get_units=False,
            verbose=False,
        )
    for i in range(num_tests):
        start = time.time()
        sim.reset_simulation(num_steps_per_test, 1, )
        setup = time.time()
        while not sim.is_done():
            sim.do_simulation_step(np.array([1.0, 1.0]))
        end = time.time()
        times[i] = [setup - start, end - setup]

    global_end = time.time()
    print(f"Took {global_end - global_start} seconds")
    print(f"Took {num_tests * num_steps_per_test} steps")
    print(f"Average setup time: {np.mean(times[:, 0])}")
    print(f"Average simulation time: {np.mean(times[:, 1])}")
    print(f"Time per step: {(global_end - global_start) / (num_tests * num_steps_per_test)}")
