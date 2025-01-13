import os
import subprocess

from matplotlib import pyplot as plt
from mushroom_rl.core import Core


class BetterMujocoCore(Core):
    """
    Class to extend the Core class from MushroomRL to enable custom rendering for Mujoco environments.

    This class is a subclass of the Core class from MushroomRL. It extends the class to enable custom rendering for Mujoco environments.
    The custom rendering is done by saving the frames of the environment as png files in a temporary directory.
    The frames can then be converted to a video using ffmpeg.
    """

    def __init__(self, agent, mdp, callbacks_fit=None, callback_step=None, custom_rendering_enabled=True):
        super().__init__(agent, mdp, callbacks_fit, callback_step)
        self.frame_dir = "./tmp"
        self._custom_rendering_enabled = custom_rendering_enabled

        if self._custom_rendering_enabled:
            try:
                subprocess.run(["ffmpeg", "-version"], check=True)
                print("FFmpeg found. Custom rendering enabled.")
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError("ffmpeg is required for custom rendering. Please install it and make sure it is in your PATH or set custom_rendering_enabled=False.")

        if not os.path.exists(self.frame_dir):
            os.makedirs(self.frame_dir)

    def _step(self, render):
        """
        Override the _step method to save frames as png files if custom rendering is enabled.
        """
        results = super()._step(render and not self._custom_rendering_enabled)

        if render and self._custom_rendering_enabled:
            plt.imsave(f"./tmp/frame_{self._episode_steps:04d}.png", self.mdp.env.physics.render())

        return results

    def render_episode(self, fname, custom_tmp_path=None):
        """
        Render an episode as a video.

        :param fname: The filename of the video.
        :param custom_tmp_path: The path to the temporary directory where the frames are saved.
        :return: None
        """
        if not self._custom_rendering_enabled:
            raise ValueError("Custom rendering is not enabled. Set custom_rendering_enabled to True in the core constructor.")

        if custom_tmp_path is not None:
            if not os.path.exists(custom_tmp_path):
                os.makedirs(custom_tmp_path)
                self.frame_dir = custom_tmp_path
        self.evaluate(n_episodes=1, render=True, quiet=True)
        cmd = f"ffmpeg -y -r 50 -i {self.frame_dir}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p {fname} > /dev/null 2>&1"
        os.system(cmd)

    def clear_render_cache(self):
        """
        Clear the temporary directory where the frames are saved.
        """
        for file in os.listdir(self.frame_dir):
            os.remove(os.path.join(self.frame_dir, file))