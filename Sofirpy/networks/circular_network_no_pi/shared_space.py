import numpy as np

from Sofirpy.networks.circular_network.shared_space import SharedSpace


class SharedSpaceWithoutPi(SharedSpace):
    def step(self, time: float, action: np.ndarray):
        """Performs the recurring actions of the different agent types based on the perception of its environment.

        Accepts the chosen actions for either just the pump agents or all agents.
        The actions must be given as an array:
        [0:2] Pump speeds for the pump agents (0.0 to 1.0)
        [2:6] Demand volume flows for the consumer agents (m^3/h) (optional)

        Args:
            time (float): Time step of the co-simulation in seconds.
            action (np.ndarray): Array with the chosen actions for the pump agents.
        """

        # for t=0.0 FMU returns only none values, that is why it is not necessary to update
        # before the first iteration

        if self.time_index % self.control_step_interval == 0:
            for agent in self.all_agents:
                agent.write_FMU_data()

            for idx, consumer in enumerate(self.consumer_agents):
                if idx < len(action):
                    consumer.set_action(float(action[idx]))
                else:
                    consumer.set_action(0.0)

            for idx, pump in enumerate(self.pump_agents):
                chosen_speed = .65
                pump.set_action(chosen_speed)
            self.time_index = 0

        else:
            for agent in self.all_agents:
                agent.set_action(agent.old_action)

            self.time_index += 1