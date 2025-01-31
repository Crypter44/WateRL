import datetime
import json
import pickle
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class Serializable(object):
    """
    Interface to implement serialization of a MushroomRL object.
    This provide load and save functionality to save the object in a zip file.
    It is possible to save the state of the agent with different levels of

    """

    def save(self, path, full_save=False):
        """
        Serialize and save the object to the given path on disk.

        Args:
            path (Path, str): Relative or absolute path to the object save
                location;
            full_save (bool): Flag to specify the amount of data to save for
                MushroomRL data structures.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with ZipFile(path, "w") as zip_file:
            self.save_zip(zip_file, full_save)

    def save_zip(self, zip_file, full_save, folder=""):
        """
        Serialize and save the agent to the given path on disk.

        Args:
            zip_file (ZipFile): ZipFile where te object needs to be saved;
            full_save (bool): flag to specify the amount of data to save for
                MushroomRL data structures;
            folder (string, ''): subfolder to be used by the save method.

        If a "!" character is added at the end of the method, the attribute will
        be saved only if full_save is set to True.
        """
        primitive_dictionary = dict()

        for att, method in self._save_attributes.items():

            if not method.endswith("!") or full_save:
                method = method[:-1] if method.endswith("!") else method
                attribute = getattr(self, att) if hasattr(self, att) else None

                if attribute is not None:
                    if method == "primitive":
                        primitive_dictionary[att] = attribute
                    elif method == "none":
                        pass
                    elif hasattr(self, "_save_{}".format(method)):
                        save_method = getattr(self, "_save_{}".format(method))
                        file_name = "{}.{}".format(att, method)
                        save_method(
                            zip_file,
                            file_name,
                            attribute,
                            full_save=full_save,
                            folder=folder,
                        )
                    else:
                        raise NotImplementedError(
                            "Method _save_{} is not implemented for class '{}'".format(
                                method, self.__class__.__name__
                            )
                        )

        config_data = dict(
            type=type(self),
            save_attributes=self._save_attributes,
            primitive_dictionary=primitive_dictionary,
        )

        self._save_pickle(zip_file, "config", config_data, folder=folder)

    @classmethod
    def load(cls, path):
        """
        Load and deserialize the agent from the given location on disk.

        Args:
            path (Path, string): Relative or absolute path to the agents save
                location.

        Returns:
            The loaded agent.

        """
        path = Path(path)
        if not path.exists():
            raise ValueError("Path to load agent is not valid")

        with ZipFile(path, "r") as zip_file:
            loaded_object = cls.load_zip(zip_file)

        return loaded_object

    @classmethod
    def load_zip(cls, zip_file, folder=""):
        config_path = Serializable._append_folder(folder, "config")

        try:
            object_type, save_attributes, primitive_dictionary = cls._load_pickle(
                zip_file, config_path
            ).values()
        except KeyError:
            return None

        if object_type is list:
            return cls._load_list(zip_file, folder, primitive_dictionary["len"])
        else:
            loaded_object = object_type.__new__(object_type)
            setattr(loaded_object, "_save_attributes", save_attributes)

            for att, method in save_attributes.items():
                mandatory = not method.endswith("!")
                method = method[:-1] if not mandatory else method
                file_name = Serializable._append_folder(
                    folder, "{}.{}".format(att, method)
                )

                if method == "primitive" and att in primitive_dictionary:
                    setattr(loaded_object, att, primitive_dictionary[att])
                elif file_name in zip_file.namelist() or (
                    method == "mushroom" and mandatory
                ):
                    load_method = getattr(cls, "_load_{}".format(method))
                    if load_method is None:
                        raise NotImplementedError(
                            "Method _load_{} is not" "implemented".format(method)
                        )
                    att_val = load_method(zip_file, file_name)
                    setattr(loaded_object, att, att_val)

                else:
                    setattr(loaded_object, att, None)

            loaded_object._post_load()

            return loaded_object

    @classmethod
    def _load_list(self, zip_file, folder, length):
        loaded_list = list()

        for i in range(length):
            element_folder = Serializable._append_folder(folder, str(i))
            loaded_element = Serializable.load_zip(zip_file, element_folder)
            loaded_list.append(loaded_element)

        return loaded_list

    def copy(self):
        """
        Returns:
             A deepcopy of the agent.

        """
        return deepcopy(self)

    def _add_save_attr(self, **attr_dict):
        """
        Add attributes that should be saved for an agent.
        For every attribute, it is necessary to specify the method to be used to
        save and load.
        Available methods are: numpy, mushroom, torch, json, pickle, primitive
        and none. The primitive method can be used to store primitive attributes,
        while the none method always skip the attribute, but ensure that it is
        initialized to None after the load. The mushroom method can be used with
        classes that implement the Serializable interface. All the other methods
        use the library named.
        If a "!" character is added at the end of the method, the field will be
        saved only if full_save is set to True.

        Args:
            **attr_dict: dictionary of attributes mapped to the method
                that should be used to save and load them.

        """
        if not hasattr(self, "_save_attributes"):
            self._save_attributes = dict()
        self._save_attributes.update(attr_dict)

    def _post_load(self):
        """
        This method can be overwritten to implement logic that is executed
        after the loading of the agent.

        """
        pass

    @staticmethod
    def _append_folder(folder, name):
        if folder:
            return folder + "/" + name
        else:
            return name

    @staticmethod
    def _load_pickle(zip_file, name):
        with zip_file.open(name, "r") as f:
            return pickle.load(f)

    @staticmethod
    def _load_numpy(zip_file, name):
        with zip_file.open(name, "r") as f:
            return np.load(f)

    @staticmethod
    def _load_torch(zip_file, name):
        with zip_file.open(name, "r") as f:
            return torch.load(f)

    @staticmethod
    def _load_json(zip_file, name):
        with zip_file.open(name, "r") as f:
            return json.load(f)

    @staticmethod
    def _load_mushroom(zip_file, name):
        return Serializable.load_zip(zip_file, name)

    @staticmethod
    def _save_pickle(zip_file, name, obj, folder, **_):
        path = Serializable._append_folder(folder, name)
        with zip_file.open(path, "w") as f:
            pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)

    @staticmethod
    def _save_numpy(zip_file, name, obj, folder, **_):
        path = Serializable._append_folder(folder, name)
        with zip_file.open(path, "w") as f:
            np.save(f, obj)

    @staticmethod
    def _save_torch(zip_file, name, obj, folder, **_):
        path = Serializable._append_folder(folder, name)
        with zip_file.open(path, "w") as f:
            torch.save(obj, f)

    @staticmethod
    def _save_json(zip_file, name, obj, folder, **_):
        path = Serializable._append_folder(folder, name)
        with zip_file.open(path, "w") as f:
            string = json.dumps(obj)
            f.write(string.encode("utf8"))

    @staticmethod
    def _save_mushroom(zip_file, name, obj, folder, full_save):
        new_folder = Serializable._append_folder(folder, name)
        if isinstance(obj, list):
            config_data = dict(
                type=list,
                save_attributes=dict(),
                primitive_dictionary=dict(len=len(obj)),
            )

            Serializable._save_pickle(
                zip_file, "config", config_data, folder=new_folder
            )
            for i, element in enumerate(obj):
                element_folder = Serializable._append_folder(new_folder, str(i))
                element.save_zip(zip_file, full_save=full_save, folder=element_folder)
        else:
            obj.save_zip(zip_file, full_save=full_save, folder=new_folder)

    @staticmethod
    def _get_serialization_method(class_name):
        if issubclass(class_name, Serializable):
            return "mushroom"
        else:
            return "pickle"


class MDPInfo(Serializable):
    """
    This class is used to store the information of the environment.

    """

    def __init__(
        self,
        state_space,
        observation_space,
        action_space,
        discrete_actions,
        gamma,
        horizon=None,
        dt=1e-1,
        has_obs=False,
        has_action_masks=False,
        n_agents=1,
    ):
        """
        Constructor.

        Args:
             observation_space ([Box, Discrete]): the state space;
             action_space ([list, Box, Discrete]): the action spaces of the agents;
             gamma (float): the discount factor;
             horizon (int): the horizon.

        """
        self.state_space = state_space
        self.observation_space = observation_space
        self.action_space = action_space
        self.discrete_actions = discrete_actions
        self.gamma = gamma
        self.horizon = horizon
        self.dt = dt
        self.has_obs = has_obs
        self.has_action_masks = has_action_masks
        self.n_agents = n_agents

        self._add_save_attr(
            state_space="mushroom",
            observation_space="mushroom",
            action_space="mushroom",
            gamma="primitive",
            horizon="primitive",
            dt="primitive",
            has_obs="primitive",
            has_action_masks="primitive",
            n_agents="primitive",
        )

    @property
    def size(self):
        """
        Returns:
            The sum of the number of discrete states and discrete actions. Only
            works for discrete spaces.

        """
        return (
            self.observation_space.size
            + self.action_space[0].size
            + self.action_space[1].size
        )

    @property
    def shape(self):
        """
        Returns:
            The concatenation of the shape tuple of the state and action
            spaces.

        """
        return (
            self.observation_space.shape
            + self.action_space[0].shape
            + self.action_space[1].shape
        )


class Environment(Serializable):
    """
    Basic interface used by any multi agent environment.

    """

    def __init__(self, mdp_info):
        """
        Constructor.

        Args:
             mdp_info (MDPInfo): an object containing the info of the
                environment.

        """
        self._mdp_info = mdp_info

        self._add_save_attr(_mdp_info="mushroom")

    def seed(self, seed):
        """
        Set the seed of the environment.

        Args:
            seed (float): the value of the seed.

        """
        self.env.seed(seed)

    def reset(self):
        """
        Reset the current state.

        Returns:
            The current state.

        """
        raise NotImplementedError

    def step(self, actions):
        """
        Move the agent from its current state according to the action.

        Args:
            action, list[np.ndarray]: the list of actions to execute.

        Returns:
            The list of observations reached by the agent executing actions in their current
            state, the next state of the environment the rewards obtained in the transition and a flag to signal
            if the next state is absorbing. Also an additional info dictionary is
            returned (possibly empty).

        """
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def stop(self):
        """
        Method used to stop an mdp. Useful when dealing with real world
        environments, simulators, or when using openai-gym rendering

        """
        pass

    @property
    def info(self):
        """
        Returns:
             An object containing the info of the environment.

        """
        return self._mdp_info

    @staticmethod
    def _bound(x, min_value, max_value):
        """
        Method used to bound state and action variables.

        Args:
            x: the variable to bound;
            min_value: the minimum value;
            max_value: the maximum value;

        Returns:
            The bounded variable.

        """
        return np.maximum(min_value, np.minimum(x, max_value))


class VideoRecorder(object):
    """
    Simple video record that creates a video from a stream of images.
    """

    def __init__(
        self, path="./mushroom_rl_recordings", tag=None, video_name=None, fps=60
    ):
        """
        Constructor.

        Args:
            path: Path at which videos will be stored.
            tag: Name of the directory at path in which the video will be stored. If None, a timestamp will be created.
            video_name: Name of the video without extension. Default is "recording".
            fps: Frame rate of the video.
        """

        if tag is None:
            date_time = datetime.datetime.now()
            tag = date_time.strftime("%d-%m-%Y_%H-%M-%S")

        self._path = Path(path)
        self._path = self._path / tag

        self._video_name = video_name if video_name else "recording"
        self._counter = 0

        self._fps = fps

        self._video_writer = None

    def __call__(self, frame):
        """
        Args:
            frame (np.ndarray): Frame to be added to the video (H, W, RGB)
        """
        assert frame is not None

        if self._video_writer is None:
            height, width = frame.shape[:2]
            self._create_video_writer(height, width)

        self._video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def _create_video_writer(self, height, width):

        name = self._video_name
        if self._counter > 0:
            name += f"-{self._counter}.mp4"
        else:
            name += ".mp4"

        self._path.mkdir(parents=True, exist_ok=True)

        path = self._path / name

        self._video_writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            self._fps,
            (width, height),
        )

    def stop(self):
        cv2.destroyAllWindows()
        self._video_writer.release()
        self._video_writer = None
        self._counter += 1


class MultiAgentCore(object):
    def __init__(
        self,
        agents: list = None,
        mdp: Environment = None,
        callbacks_step: dict = {},
    ):
        """
        Constructor.

        Args:
            agent (Agent): list of agents moving according to a policy;
            mdp (Environment): the environment in which the agent moves;
            callbacks_step (dict): dict of callbacks to execute after each step
        """
        self.agents = agents
        self.mdp = mdp
        self.callbacks_step = callbacks_step
        self.obs_last_action = getattr(agents[0], "_obs_last_action", False)

        self._state = None
        self._obs = None
        self._action_masks = None

        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter_per_agent = np.zeros(
            self.mdp.info.n_agents, dtype=int
        )
        self._current_steps_counter_per_agent = np.zeros(
            self.mdp.info.n_agents, dtype=int
        )
        self._episode_steps = None

        self._record = self._build_recorder_class()

    def learn(
        self,
        n_steps=None,
        n_episodes=None,
        n_steps_per_fit_per_agent=None,
        n_episodes_per_fit_per_agent=None,
        render=False,
        quiet=False,
    ):
        """
        This function moves the agent in the environment and fits the policy
        using the collected samples. The agent can be moved for a given number
        of steps or a given number of episodes and, independently from this
        choice, the policy can be fitted after a given number of steps or a
        given number of episodes. By default, the environment is reset.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            n_steps_per_fit_per_agent (list, None): number of steps between each fit of each agent's
                policies;
            n_episodes_per_fit_per_agent (list, None): number of episodes between each fit
                of each agent's policies;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        assert (
            n_episodes_per_fit_per_agent is not None
            and n_steps_per_fit_per_agent is None
        ) or (
            n_episodes_per_fit_per_agent is None
            and n_steps_per_fit_per_agent is not None
        )

        self._n_steps_per_fit_per_agent = n_steps_per_fit_per_agent
        self._n_episodes_per_fit_per_agent = n_episodes_per_fit_per_agent

        fit_condition_per_agent = list()
        if n_steps_per_fit_per_agent is not None:  # train every no. steps
            for idx_agent_loop, n_steps_per_fit_loop in enumerate(n_steps_per_fit_per_agent):
                fit_condition_per_agent.append(
                    lambda idx_agent=idx_agent_loop, n_steps_per_fit=n_steps_per_fit_loop: self._current_steps_counter_per_agent[
                        idx_agent
                    ]
                    >= n_steps_per_fit
                )
        else:  # train every no. episodes
            for idx_agent_loop, n_episodes_per_fit_loop in enumerate(
                self._n_episodes_per_fit_per_agent
            ):
                fit_condition_per_agent.append(
                    lambda idx_agent=idx_agent_loop, n_episodes_per_fit=n_episodes_per_fit_loop: self._current_episodes_counter_per_agent[
                        idx_agent
                    ]
                    >= n_episodes_per_fit
                )

        need_complete_episodes = (
            True if n_episodes_per_fit_per_agent is not None else False
        )

        self._run(
            n_steps,
            n_episodes,
            fit_condition_per_agent,
            render,
            quiet,
            need_complete_episodes=need_complete_episodes,
        )

    def evaluate(
        self,
        n_steps=None,
        n_episodes=None,
        render=False,
        quiet=False,
        record=False,
    ):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from
        a set of initial states for the whole episode. By default, the
        environment is reset.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        assert (render and record) or (
            not record
        ), "To record, the render flag must be set to true"

        fit_condition_per_agent = []
        for _ in range(self.mdp.info.n_agents):
            fit_condition_per_agent.append(lambda: False)

        return self._run(
            n_steps,
            n_episodes,
            fit_condition_per_agent,
            render,
            quiet,
            record,
            need_complete_episodes=False,
        )

    def _run(
        self,
        n_steps,
        n_episodes,
        fit_condition_per_agent,
        render,
        quiet,
        record=False,
        need_complete_episodes=False,
    ):
        assert (
            n_episodes is not None
            and n_steps is None
            or n_episodes is None
            and n_steps is not None
        )

        if n_steps is not None:
            move_condition = lambda: self._total_steps_counter < n_steps

            steps_progress_bar = tqdm(
                total=n_steps, dynamic_ncols=True, disable=quiet, leave=False
            )
            episodes_progress_bar = tqdm(disable=True)
        else:
            move_condition = lambda: self._total_episodes_counter < n_episodes

            steps_progress_bar = tqdm(disable=True)
            episodes_progress_bar = tqdm(
                total=n_episodes, dynamic_ncols=True, disable=quiet, leave=False
            )

        dataset, dataset_info = self._run_impl(
            move_condition,
            fit_condition_per_agent,
            steps_progress_bar,
            episodes_progress_bar,
            render,
            record,
            need_complete_episodes,
        )
        return dataset, dataset_info

    def _run_impl(
        self,
        move_condition,
        fit_condition_per_agent,
        steps_progress_bar,
        episodes_progress_bar,
        render,
        record,
        need_complete_episodes,
    ):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter_per_agent = np.zeros(
            self.mdp.info.n_agents, dtype=int
        )
        self._current_steps_counter_per_agent = np.zeros(
            self.mdp.info.n_agents, dtype=int
        )

        dataset = []
        dataset_info = []
        last = True

        if need_complete_episodes:
            step_count_move_condition = move_condition
            move_condition = lambda: step_count_move_condition() or not last

        while move_condition():
            if last:
                self.reset()

            sample, info = self._step(render, record)

            self._total_steps_counter += 1
            self._current_steps_counter_per_agent += 1
            steps_progress_bar.update(1)

            last = sample["last"]
            if last:
                self._total_episodes_counter += 1
                self._current_episodes_counter_per_agent += 1
                episodes_progress_bar.update(1)

            dataset.append(sample)
            dataset_info.append(info)

            for idx_agent, fit_condition in enumerate(fit_condition_per_agent):
                actor_losses = []
                critic_losses = []
                if fit_condition():
                    actor_loss, critic_loss = self.agents[idx_agent].fit(dataset)
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    self._current_episodes_counter_per_agent[idx_agent] = 0
                    self._current_steps_counter_per_agent[idx_agent] = 0
                    # Only save loss info after all agents have been fit
                    if idx_agent == len(fit_condition_per_agent) - 1:
                        info["actor_loss"] = np.mean(actor_losses)
                        info["critic_loss"] = np.mean(critic_losses)
                if not any(
                    self._current_steps_counter_per_agent
                ):  # all agents have been fit
                    dataset = list()
                    dataset_info = list()

            self._get_callbacks(sample, info)

        for agent in self.agents:
            agent.stop()
        self.mdp.stop()
        if record:
            self._record.stop()

        return dataset, dataset_info

    def _step(self, render, record):
        actions = self._get_actions()

        step = self.mdp.step(actions)

        next_state = step["state"]
        next_obs = step.get("obs", None)
        rewards = step["rewards"]
        absorbing = step["absorbing"]
        next_action_masks = step.get("action_masks", None)
        info = step.get("info", None)

        self._episode_steps += 1

        if self.obs_last_action:
            for i, step_obs in enumerate(next_obs):
                if self.mdp.info.discrete_actions:  # make integer action one-hot
                    action = np.zeros(self.mdp.info.action_space[i].n)
                    action[actions[i]] = 1
                else:
                    action = actions[i]
                next_obs[i] = np.concatenate([step_obs, action])

        if render:
            render_info = {}
            frame = self.mdp.render(render_info)

            if record:
                self._record(frame)

        last = (self._episode_steps >= self.mdp.info.horizon) or absorbing

        obs = self._obs
        next_obs = next_obs.copy()
        self._obs = next_obs

        state = self._state
        next_state = next_state.copy()
        self._state = next_state

        action_masks = self._action_masks.copy() if self._action_masks is not None else None
        self._action_masks = next_action_masks

        sample = {
            "state": state,
            "obs": obs,
            "action_masks": action_masks,
            "actions": actions,
            "rewards": rewards,
            "next_state": next_state,
            "next_obs": next_obs,
            "next_action_masks": next_action_masks,
            "absorbing": absorbing,
            "last": last,
        }

        return sample, info

    def _get_actions(self):
        actions = list()
        for idx_agent in range(self.mdp.info.n_agents):
            if self.mdp.info.has_obs:
                if self.mdp.info.has_action_masks:
                    actions.append(
                        self.agents[idx_agent].draw_action(
                            self._obs[idx_agent], self._action_masks[idx_agent]
                        )
                    )
                else:
                    actions.append(
                        self.agents[idx_agent].draw_action(self._obs[idx_agent])
                    )
            else:
                actions.append(self.agents[idx_agent].draw_action(self._state))
        return actions

    def _get_callbacks(self, sample, info):
        for _, callback in self.callbacks_step.items():
            callback_info = {
                "sample": sample,
                "info": info,
            }
            callback(callback_info)

    def reset(self):
        """
        Reset the state of the mdp and agents.

        """
        init_step = self.mdp.reset()

        self._state = init_step["state"]
        self._obs = init_step.get("obs", None)
        self._action_masks = init_step.get("action_masks", None)

        if self.obs_last_action:
            for i, init_obs in enumerate(self._obs):
                if self.mdp.info.discrete_actions:  # make integer action one-hot
                    action = np.zeros(self.mdp.info.action_space[i].n)
                else:
                    action = np.zeros(self.mdp.info.action_space[i].shape)
                self._obs[i] = np.concatenate([init_obs, action])

        for agent in self.agents:
            agent.episode_start()
            agent.next_action = None
        self._episode_steps = 0

    def set_random_mode(self):
        for agent in self.agents:
            agent.set_random_mode()

    def set_training_mode(self):
        for agent in self.agents:
            agent.set_training_mode()

    def set_testing_mode(self):
        for agent in self.agents:
            agent.set_testing_mode()

    def set_callbacks_step(self, name, callback):
        self.callbacks_step[name] = callback

    def remove_callbacks_step(self, name):
        self.callbacks_step.pop(name)

    def clear_callbacks_step(self, name):
        self.callbacks_step[name].clear()

    def clear_all_callbacks_step(self):
        for callback in self.callbacks_step.values():
            callback.clear()

    def set_memory_profiler(self, memory_profiler):
        self.memory_profiler = memory_profiler

    def _build_recorder_class(self, recorder_class=None, fps=None, **kwargs):
        """
        Method to create a video recorder class.

        Args:
            recorder_class (class): the class used to record the video. By default, we use the ``VideoRecorder`` class
                from mushroom. The class must implement the ``__call__`` and ``stop`` methods.

        Returns:
             The recorder object.

        """

        if not recorder_class:
            recorder_class = VideoRecorder

        if not fps:
            fps = int(1 / self.mdp.info.dt)

        return recorder_class(fps=fps, **kwargs)


class ContinuousActorNetwork(nn.Module):
    """
    Generic continuous actor network architecture
    """

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        n_features_h1 = n_features[0]
        n_features_h2 = n_features[1]

        self._h1 = nn.Linear(n_input, n_features_h1)
        self._h2 = nn.Linear(n_features_h1, n_features_h2)
        self._out = nn.Linear(n_features_h2, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._out.weight, gain=nn.init.calculate_gain("tanh"))

    def forward(self, state):
        features1 = F.relu(self._h1(state.float()))
        features2 = F.relu(self._h2(features1))
        actions = F.tanh(self._out(features2))

        return actions


class ContinuousCriticNetwork(nn.Module):
    """
    Generic continuous critic network architecture
    """

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        n_features_h1 = n_features[0]
        n_features_h2 = n_features[1]

        self._h1 = nn.Linear(n_input, n_features_h1)
        self._h2 = nn.Linear(n_features_h1, n_features_h2)
        self._out = nn.Linear(n_features_h2, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._out.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=-1)

        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._out(features2)

        return torch.squeeze(q)


class Policy(Serializable):
    """
    Interface representing a generic policy.
    A policy is a probability distribution that gives the probability of taking
    an action given a specified state.
    A policy is used by mushroom agents to interact with the environment.

    """

    def __call__(self, *args):
        """
        Compute the probability of taking action in a certain state following
        the policy.

        Args:
            *args (list): list containing a state or a state and an action.

        Returns:
            The probability of all actions following the policy in the given
            state if the list contains only the state, else the probability
            of the given action in the given state following the policy. If
            the action space is continuous, state and action must be provided

        """
        raise NotImplementedError

    def draw_action(self, state):
        """
        Sample an action in ``state`` using the policy.

        Args:
            state (np.ndarray): the state where the agent is.

        Returns:
            The action sampled from the policy.

        """
        raise NotImplementedError

    def reset(self):
        """
        Useful when the policy needs a special initialization at the beginning
        of an episode.

        """
        pass

    def set_mode(self, mode):
        self._mode = mode


class ParametricPolicy(Policy):
    """
    Interface for a generic parametric policy.
    A parametric policy is a policy that depends on set of parameters,
    called the policy weights.
    If the policy is differentiable, the derivative of the probability for a
    specified state-action pair can be provided.
    """

    def __init__(self):
        self._add_save_attr(_approximator="mushroom")

    def diff_log(self, state, action):
        """
        Compute the gradient of the logarithm of the probability density
        function, in the specified state and action pair, i.e.:

        .. math::
            \\nabla_{\\theta}\\log p(s,a)


        Args:
            state (np.ndarray): the state where the gradient is computed
            action (np.ndarray): the action where the gradient is computed

        Returns:
            The gradient of the logarithm of the pdf w.r.t. the policy weights
        """
        raise RuntimeError("The policy is not differentiable")

    def diff(self, state, action):
        """
        Compute the derivative of the probability density function, in the
        specified state and action pair. Normally it is computed w.r.t. the
        derivative of the logarithm of the probability density function,
        exploiting the likelihood ratio trick, i.e.:

        .. math::
            \\nabla_{\\theta}p(s,a)=p(s,a)\\nabla_{\\theta}\\log p(s,a)


        Args:
            state (np.ndarray): the state where the derivative is computed
            action (np.ndarray): the action where the derivative is computed

        Returns:
            The derivative w.r.t. the  policy weights
        """
        return self(state, action) * self.diff_log(state, action)

    def set_approximator(self, approximator):
        self._approximator = approximator

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by
                the policy.
        """
        self._approximator.set_weights(weights)

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        """
        Property.

        Returns:
             The size of the policy weights.

        """
        raise NotImplementedError


class GaussianPolicy(ParametricPolicy):
    """
    This policy is commonly used in the Deep Deterministic Policy Gradient
    algorithm.
    """

    def __init__(self, sigma, action_space):
        """
        Constructor.

        Args:
            sigma (np.ndarray): sigma of the Gaussian noise
            shape (tuple): shape of the action space
        """
        super().__init__()

        self._sigma = sigma
        self._action_space = action_space
        self._mode = "train"  # options are random, train, test

        self._add_save_attr(
            _sigma="primitive",
            _action_space="mushroom",
            _mode="primitive",
        )

    def draw_action(self, state):
        if self._mode == "random":
            return self._action_space.sample()
        else:
            mu = self._approximator.predict(state).squeeze()
            if self._mode == "test":
                return mu
            elif self._mode == "train":
                noise = np.random.normal(
                    scale=self._sigma, size=self._action_space.shape
                )
                return mu + noise
            else:
                raise ValueError("Invalid policy mode given")


def compute_J_all_agents(dataset, gamma=1.0):
    """
    Compute the cumulative discounted reward of each episode for each agent in the dataset.

    Args:
        dataset (list): the dataset to consider;
        gamma (float, 1.): discount factor.

    Returns:
        The cumulative discounted reward of each episode in the dataset.

    """
    num_agents = len(dataset[0]["actions"])
    js_all_agents = list()
    for idx_agent in range(num_agents):
        js_agent = list()
        j = 0.0
        episode_steps = 0
        for i in range(len(dataset)):
            j += gamma**episode_steps * dataset[i]["rewards"][idx_agent]
            episode_steps += 1
            if dataset[i]["last"] or i == len(dataset) - 1:
                js_agent.append(j)
                j = 0.0
                episode_steps = 0
        if len(js_agent) == 0:
            js_agent.append(0.0)
        js_all_agents.append(js_agent)

    return js_all_agents


class ReplayMemory(Serializable):
    def __init__(self, max_size, state_dim, action_dim, discrete_actions=False):
        """
        Constructor.

        Args:
            max_size (int): maximum number of elements that the replay memory
                can contain.
            state_dim (int): dimension of the state space.
            action_dim (int): dimension of the action space.
            discrete_actions (bool): whether the action space is discrete or not.
        """
        self._max_size = max_size
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._discrete_actions = discrete_actions

        self.reset()

        self._add_save_attr(
            _max_size="primitive",
            _state_dim="primitive",
            _action_dim="primitive",
            _discrete_actions="primitive",
            _idx="primitive",
            _full="primitive",
            _states="numpy",
            _actions="numpy",
            _rewards="numpy",
            _next_states="numpy",
            _absorbing="numpy",
            _last="numpy",
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;

        """
        for sample in dataset:
            self._states[self._idx] = sample["state"]
            self._actions[self._idx] = sample["action"]
            self._rewards[self._idx] = sample["reward"]
            self._next_states[self._idx] = sample["next_state"]
            self._absorbing[self._idx] = sample["absorbing"]
            self._last[self._idx] = sample["last"]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        """
        Returns the provided number of samples from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        indices = np.random.randint(self.size, size=n_samples)

        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        absorbing = self._absorbing[indices]
        last = self._last[indices]

        return states, actions, rewards, next_states, absorbing, last

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = np.empty((self._max_size, self._state_dim), dtype=np.float32)
        if self._discrete_actions:
            self._actions = np.empty((self._max_size, 1), dtype=np.int32)
        else:
            self._actions = np.empty(
                (self._max_size, self._action_dim), dtype=np.float32
            )
        self._rewards = np.empty(self._max_size, dtype=np.float32)
        self._next_states = np.empty(
            (self._max_size, self._state_dim), dtype=np.float32
        )
        self._absorbing = np.empty(self._max_size, dtype=bool)
        self._last = np.empty(self._max_size, dtype=bool)

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size


class ReplayMemoryObs(ReplayMemory):
    """
    Replay memory that stores observations instead of states.
    """

    def __init__(
        self, max_size, state_dim, obs_dim, action_dim, discrete_actions=False
    ):
        self._obs_dim = obs_dim
        super().__init__(max_size, state_dim, action_dim, discrete_actions)

        self._add_save_attr(
            _obs_dim="primitive",
            _obs="numpy",
            _next_obs="numpy",
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;

        """
        for sample in dataset:
            self._states[self._idx] = sample["state"]
            self._obs[self._idx] = sample["obs"]
            self._actions[self._idx] = sample["action"]
            self._rewards[self._idx] = sample["reward"]
            self._next_states[self._idx] = sample["next_state"]
            self._next_obs[self._idx] = sample["next_obs"]
            self._absorbing[self._idx] = sample["absorbing"]
            self._last[self._idx] = sample["last"]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        """
        Returns the provided number of samples from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        indices = np.random.randint(self.size, size=n_samples)

        states = self._states[indices]
        obs = self._obs[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        next_obs = self._next_obs[indices]
        absorbing = self._absorbing[indices]
        last = self._last[indices]

        return states, obs, actions, rewards, next_states, next_obs, absorbing, last

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = np.empty((self._max_size, self._state_dim), dtype=np.float32)
        self._obs = np.empty((self._max_size, self._obs_dim), dtype=np.float32)
        if self._discrete_actions:
            self._actions = np.empty((self._max_size, 1), dtype=np.int32)
        else:
            self._actions = np.empty(
                (self._max_size, self._action_dim), dtype=np.float32
            )
        self._rewards = np.empty(self._max_size, dtype=np.float32)
        self._next_states = np.empty(
            (self._max_size, self._state_dim), dtype=np.float32
        )
        self._next_obs = np.empty((self._max_size, self._obs_dim), dtype=np.float32)
        self._absorbing = np.empty(self._max_size, dtype=bool)
        self._last = np.empty(self._max_size, dtype=bool)


class Box(Serializable):
    """
    This class implements functions to manage continuous states and action
    spaces. It is similar to the ``Box`` class in ``gym.spaces.box``.

    """

    def __init__(self, low, high, shape=None):
        """
        Constructor.

        Args:
            low ([float, np.ndarray]): the minimum value of each dimension of
                the space. If a scalar value is provided, this value is
                considered as the minimum one for each dimension. If a
                np.ndarray is provided, each i-th element is considered the
                minimum value of the i-th dimension;
            high ([float, np.ndarray]): the maximum value of dimensions of the
                space. If a scalar value is provided, this value is considered
                as the maximum one for each dimension. If a np.ndarray is
                provided, each i-th element is considered the maximum value
                of the i-th dimension;
            shape (np.ndarray, None): the dimension of the space. Must match
                the shape of ``low`` and ``high``, if they are np.ndarray.

        """

        self._low = low
        self._high = high

        if shape is None:
            self._shape = low.shape
        else:
            self._shape = shape
            if np.isscalar(low) and np.isscalar(high):
                self._low += np.zeros(shape)
                self._high += np.zeros(shape)

        assert self._low.shape == self._high.shape

        self._add_save_attr(_low="numpy", _high="numpy")

    def sample(self):
        """
        Returns:
            A random sample from the space.
        """
        return np.random.uniform(self.low, self.high)

    @property
    def low(self):
        """
        Returns:
             The minimum value of each dimension of the space.

        """
        return self._low

    @property
    def high(self):
        """
        Returns:
             The maximum value of each dimension of the space.

        """
        return self._high

    @property
    def shape(self):
        """
        Returns:
            The dimensions of the space.

        """
        return self._shape

    def _post_load(self):
        self._shape = self._low.shape



class Agent(Serializable):
    """
    This class implements the functions to manage the agent (e.g. move the agent
    following its policy).

    """

    def __init__(self, mdp_info, policy, idx_agent):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            policy (Policy): the policy followed by the agent;
            features (object, None): features to extract from the state.

        """
        self.mdp_info = mdp_info
        self.policy = policy
        self._idx_agent = idx_agent

        self._add_save_attr(
            mdp_info="mushroom",
            policy="mushroom",
            _idx_agent="primitive",
        )

    def fit(self, dataset):
        """
        Fit step.

        Args:
            dataset (list): the dataset.

        """
        raise NotImplementedError("Agent is an abstract class")

    def draw_action(self, state):
        """
        Return the action to execute in the given state. It is the action
        returned by the policy or the action set by the algorithm (e.g. in the
        case of SARSA).

        Args:
            state (np.ndarray): the state where the agent is.

        Returns:
            The action to be executed.

        """
        return self.policy.draw_action(state)

    def episode_start(self):
        """
        Called by the agent when a new episode starts.

        """
        self.policy.reset()

    def stop(self):
        """
        Method used to stop an agent. Useful when dealing with real world
        environments, simulators, or to cleanup environments internals after
        a core learn/evaluate to enforce consistency.

        """
        pass

    def set_logger(self, logger):
        """
        Setter that can be used to pass a logger to the algorithm

        Args:
            logger (Logger): the logger to be used by the algorithm.

        """
        self._logger = logger

    def set_profiler(self, profiler):
        self.profiler = profiler

    def set_random_mode(self):
        self.policy.set_mode("random")

    def set_training_mode(self):
        self.policy.set_mode("train")

    def set_testing_mode(self):
        self.policy.set_mode("test")


def set_weights(parameters, weights, use_cuda):
    """
    Function used to set the value of a set of torch parameters given a
    vector of values.

    Args:
        parameters (list): list of parameters to be considered;
        weights (numpy.ndarray): array of the new values for
            the parameters;
        use_cuda (bool): whether the parameters are cuda tensors or not;

    """
    idx = 0
    for p in parameters:
        shape = p.data.shape

        c = 1
        for s in shape:
            c *= s

        w = np.reshape(weights[idx:idx + c], shape)

        if not use_cuda:
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
        else:
            w_tensor = torch.from_numpy(w).type(p.data.dtype).cuda()

        p.data = w_tensor
        idx += c

    assert idx == weights.size


def get_weights(parameters):
    """
    Function used to get the value of a set of torch parameters as
    a single vector of values.

    Args:
        parameters (list): list of parameters to be considered.

    Returns:
        A numpy vector consisting of all the values of the vectors.

    """
    weights = list()

    for p in parameters:
        w = p.data.detach().cpu().numpy()
        weights.append(w.flatten())

    weights = np.concatenate(weights, 0)

    return weights


class TorchApproximator(Serializable):
    """
    Class to interface a pytorch model to the mushroom Regressor interface.
    This class implements all is needed to use a generic pytorch model and train
    it using a specified optimizer and objective function.
    This class supports also minibatches.

    """

    def __init__(
        self,
        input_shape,
        output_shape,
        network,
        optimizer,
        loss=None,
        n_features=None,
        use_cuda=False,
        **network_params
    ):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input of the network;
            output_shape (tuple): shape of the output of the network;
            network (torch.nn.Module): the network class to use;
            optimizer (dict): the optimizer used for every fit step;
            loss (torch.nn.functional): the loss function to optimize in the
                fit method;
            n_fit_targets (int, 1): the number of fit targets used by the fit
                method of the network;
            use_cuda (bool, False): if True, runs the network on the GPU;
            **network_params: dictionary of parameters needed to construct the
                network.

        """

        self._use_cuda = use_cuda

        self.network = network(
            input_shape, output_shape, n_features, use_cuda=use_cuda, **network_params
        )

        if self._use_cuda:
            self.network.cuda()

        self.optimizer = optimizer
        self._optimizer = optimizer["class"](
            self.network.parameters(), **optimizer["params"]
        )
        self._loss = loss

        self._add_save_attr(
            _use_cuda="primitive",
            network="torch",
            _optimizer="torch",
            _loss="torch",
        )

    def predict(self, *args, output_tensor=False, **kwargs):
        """
        Predict.

        Args:
            *args: input;
            output_tensor (bool, False): whether to return the output as tensor
                or not;
            **kwargs: other parameters used by the predict method
                the regressor.

        Returns:
            The predictions of the model.

        """
        if self._use_cuda:
            torch_args = [
                (
                    torch.from_numpy(x).type(torch.float32).cuda()
                    if isinstance(x, np.ndarray)
                    else x.cuda()
                )
                for x in args
            ]
        else:
            torch_args = [
                (
                    torch.from_numpy(x).type(torch.float32)
                    if isinstance(x, np.ndarray)
                    else x
                )
                for x in args
            ]
        if torch_args[0].ndim == 1:
            torch_args[0] = torch_args[0].unsqueeze(0)  # Make single state 2D
        val = self.network.forward(*torch_args, **kwargs)

        if output_tensor:
            return val
        elif isinstance(val, tuple):
            val = tuple([x.detach().numpy() for x in val])
        else:
            val = val.detach().numpy()

        return val

    def fit(self, *args, **kwargs):
        """
        Fit the model.

        Args:
            *args: input, where the last ``n_fit_targets`` elements
                are considered as the target, while the others are considered
                as input;
            **kwargs: other parameters used by the fit method of the
                regressor.

        """
        if self._use_cuda:
            torch_args = [
                torch.from_numpy(x).cuda() if isinstance(x, np.ndarray) else x.cuda()
                for x in args
            ]
        else:
            torch_args = [
                torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in args
            ]

        x = torch_args[:-1]
        y_hat = self.network(*x, **kwargs)
        y_target = torch_args[-1]

        loss = self._loss(y_hat, y_target)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def parameters(self):
        """
        Get the parameters of the network.

        Returns:
            The parameters of the network.

        """
        return list(self.network.parameters())

    def gradient(self):
        """
        Get the gradients of the network.

        Returns:
            The gradients of the network.

        """
        return torch.cat([p.grad.detach().flatten() for p in self.parameters])

    def gradient_norm(self):
        """
        Get the norm of the gradients of the network.

        Returns:
            The norm of the gradients of the network.

        """

        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in self.parameters()]),
            2.0,
        ).item()

        return total_norm

    def set_weights(self, weights):
        """
        Setter.

        Args:
            w (np.ndarray): the set of weights to set.

        """
        set_weights(self.network.parameters(), weights, self._use_cuda)

    def get_weights(self):
        """
        Getter.

        Returns:
            The set of weights of the approximator.

        """
        return get_weights(self.network.parameters())

    def set_primary_approximator(self, primary_approximator):
        """
        Setter.

        Args:
            primary_network (TorchApproximator): the primary network; take this network's
            weights.

        """
        self.network = primary_approximator.network
        self._optimizer = primary_approximator._optimizer

    @property
    def use_cuda(self):
        return self._use_cuda


class DDPG(Agent):
    """
    Deep Deterministic Policy Gradient algorithm.
    "Continuous Control with Deep Reinforcement Learning".
    Lillicrap T. P. et al.. 2016.

    """

    def __init__(
        self,
        mdp_info,
        idx_agent,
        policy,
        actor_params,
        critic_params,
        batch_size,
        target_update_frequency,
        tau,
        warmup_replay_size,
        replay_memory,
        use_cuda,
        primary_agent,
        use_mixer,
    ):
        """
        Constructor.

        """
        super().__init__(mdp_info, policy, idx_agent)

        self._batch_size = batch_size
        self._target_update_frequency = target_update_frequency
        self._tau = tau
        self._warmup_replay_size = warmup_replay_size
        self._use_mixer = use_mixer
        self._use_cuda = use_cuda

        self._replay_memory = replay_memory

        self._n_updates = 0

        self._primary_agent = primary_agent

        target_actor_params = deepcopy(actor_params)
        self.actor_approximator = TorchApproximator(**actor_params)
        self.target_actor_approximator = TorchApproximator(**target_actor_params)
        target_critic_params = deepcopy(critic_params)
        self.critic_approximator = TorchApproximator(**critic_params)
        self.target_critic_approximator = TorchApproximator(**target_critic_params)
        if primary_agent is None or idx_agent == 0:
            self._update_targets_hard()
        else:
            # Set this agent's actor, target, critic, critic target to be the same as the other agent
            self.actor_approximator.set_primary_approximator(
                primary_agent.actor_approximator
            )
            self.target_actor_approximator.set_primary_approximator(
                primary_agent.target_actor_approximator
            )
            self.critic_approximator.set_primary_approximator(
                primary_agent.critic_approximator
            )
            self.target_critic_approximator.set_primary_approximator(
                primary_agent.target_critic_approximator
            )
        self.policy.set_approximator(self.actor_approximator)
        self._optimizer = self.actor_approximator._optimizer

        self._add_save_attr(
            _batch_size="primitive",
            _target_update_frequency="primitive",
            _tau="primitive",
            _warmup_replay_size="primitive",
            _replay_memory="mushroom!",
            _n_updates="primitive",
            actor_approximator="mushroom",
            target_actor_approximator="mushroom",
            critic_approximator="mushroom",
            target_critic_approximator="mushroom",
            _optimizer="torch",
            _use_mixer="primitive",
            _use_cuda="primitive",
        )

    def draw_action(self, state, action_mask=None):
        """
        Args:
            state (np.ndarray): the state where the agent is.

        Returns:
            The action to be executed.

        """

        return self.policy.draw_action(state)

    def fit(self, dataset):
        if self._use_mixer:
            actor_loss, critic_loss = 0, 0  # storage and fitting handled by mixer
        else:
            own_dataset = self.split_dataset(dataset)
            self._replay_memory.add(own_dataset)
            actor_loss, critic_loss = self._fit()

        self._n_updates += 1
        if self._idx_agent == 0 or self._primary_agent is None:
            self._update_targets_soft()

        return actor_loss, critic_loss

    def split_dataset(self, dataset):
        own_dataset = list()
        for sample in dataset:
            own_sample = {
                "state": sample["state"],
                "obs": sample["obs"][self._idx_agent],
                "action": sample["actions"][self._idx_agent],
                "reward": sample["rewards"][self._idx_agent],
                "next_state": sample["next_state"],
                "next_obs": sample["next_obs"][self._idx_agent],
                "absorbing": sample["absorbing"],
                "last": sample["last"],
            }
            own_dataset.append(own_sample)
        return own_dataset

    def _fit(self):
        if self._replay_memory.size > self._warmup_replay_size:
            _, obs, actions, rewards, _, next_obs, absorbing, _ = (
                self._replay_memory.get(self._batch_size)
            )

            # Convert to torch tensors
            obs_t = torch.tensor(obs, dtype=torch.float32)
            actions_t = torch.tensor(actions, dtype=torch.float32)
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32)
            absorbing_t = torch.tensor(absorbing, dtype=torch.bool)

            # Critic update
            q_hat = self.critic_approximator.predict(
                obs_t, actions_t, output_tensor=True
            )
            q_next = self._next_q(next_obs_t)
            q_target = (
                rewards_t + self.mdp_info.gamma * q_next * ~absorbing_t
            ).detach()
            critic_loss = self.critic_approximator._loss(q_hat, q_target)
            critic_loss.backward()
            self.critic_approximator._optimizer.step()

            # Actor update
            loss = self._loss(obs_t)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            return loss.item(), critic_loss.item()
        else:
            return 0, 0

    def _loss(self, state):
        action = self.actor_approximator.predict(state, output_tensor=True)
        q = self.critic_approximator.predict(state, action, output_tensor=True)
        return -q.mean()

    def _draw_target_action(self, next_state_t):
        """
        Draw an action from the target actor without noise.

        Args:
            next_state_t (torch.Tensor): the state where the action is drawn.

        Returns:
            next_mu (torch.Tensor): the greedy action drawn from the target actor network.
        """

        mu_target = self.target_actor_approximator.predict(
            next_state_t, output_tensor=True
        )

        return mu_target

    def _next_q(self, next_state_t):
        """
        Args:
            next_state (torch.Tensor): the states where next action has to be
                evaluated;
            absorbing (torch.Tensor): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a_next_t = self._draw_target_action(next_state_t)
        q_next = self.target_critic_approximator.predict(
            next_state_t, a_next_t, output_tensor=True
        )

        return q_next

    def _update_targets_soft(self):
        """
        Update the target network.

        """
        self._update_target_soft(
            self.actor_approximator, self.target_actor_approximator
        )
        self._update_target_soft(
            self.critic_approximator, self.target_critic_approximator
        )

    def _update_targets_hard(self):
        """
        Update the target network.

        """
        self._update_target_hard(
            self.actor_approximator, self.target_actor_approximator
        )
        self._update_target_hard(
            self.critic_approximator, self.target_critic_approximator
        )

    def _update_target_soft(self, online, target):
        weights = self._tau * online.get_weights()
        weights += (1 - self._tau) * target.get_weights()
        target.set_weights(weights)

    def _update_target_hard(self, online, target):
        target.set_weights(online.get_weights())

    # def _post_load(self):
    #     self.actor_approximator = self.policy.approximator