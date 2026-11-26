import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import torch

import hydra
from omegaconf import OmegaConf

import calvin_env
from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id

logger = logging.getLogger(__name__)

EXPECTED_CALVIN_ENV_COMMIT = "1431a46bd36bde5903fb6345e68b5ccc30def666"
EXPECTED_CALVIN_ENV_FILES = {
    "calvin_env/envs/play_table_env.py": "dc19a40e856c99265d90c9a4ac99c58c8661b76bef68c639984ef6d80711ab60",
    "calvin_env/robot/robot.py": "65721aa5f6bbb74a7bfabd53fe05699c0fc281d682363f065600f94622a396a7",
    "calvin_env/camera/static_camera.py": "aec23cf05bdcdbc5768d6f7b0a4b8415925c3ac0b703405bc270130f11cc0dcf",
    "conf/env/play_table_env.yaml": "b72a05e08025737ca0b4e5c2c097fdbc57d0ce19e13838c9587be0b464a9e3a6",
}


class CalvinEnvWrapperRaw(gym.Wrapper):
    def __init__(self, abs_datasets_dir, observation_space, device, show_gui=False, **kwargs):
        """Environment wrapper which returns raw observations.

        Args:
            abs_datasets_dir: absolute datset directory
            observation_sapce: {'rgb_obs': ['rgb_static', 'rgb_gripper'], 'depth_obs': [], 'state_obs': ['robot_obs'], 'actions': ['rel_actions'], 'language': ['language']}
        """
        if show_gui:
            raise ValueError("CALVIN paper evaluation requires headless EGL; show_gui must be False")
        self._validate_calvin_env_revision()
        self.set_egl_device(device)
        env = self._get_env_with_egl(
            abs_datasets_dir,
            observation_space=observation_space,
            show_gui=show_gui,
            **kwargs,
        )
        super(CalvinEnvWrapperRaw, self).__init__(env)
        self.observation_space_keys = observation_space
        self.device = device
        self.relative_actions = "rel_actions" in self.observation_space_keys["actions"]
        self._last_obs = None
        logger.info(f"Initialized PlayTableEnv for device {self.device}")
        logger.info(
            "CALVIN renderer: use_egl=%s EGL_VISIBLE_DEVICES=%s PYOPENGL_PLATFORM=%s",
            getattr(self.env, "use_egl", None),
            os.environ.get("EGL_VISIBLE_DEVICES"),
            os.environ.get("PYOPENGL_PLATFORM"),
        )
        if getattr(self.env, "use_egl", None) is not True:
            raise RuntimeError("CALVIN environment was not instantiated with use_egl=True")

    @staticmethod
    def _validate_calvin_env_revision():
        """Reject known-incompatible or unidentifiable CALVIN environment code."""
        package_root = Path(calvin_env.__file__).resolve().parent
        repo_root = package_root.parent
        import hashlib

        actual_hashes = {}
        for relative_path in EXPECTED_CALVIN_ENV_FILES:
            path = repo_root / relative_path
            actual_hashes[relative_path] = (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
            )
        git_dir = repo_root / ".git"
        git_commit = None
        if git_dir.exists():
            try:
                import subprocess

                git_commit = subprocess.check_output(
                    ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                git_commit = None

        print(
            "[CALVIN] calvin_env revision: "
            f"root={repo_root}, git={git_commit or 'unknown'}, "
            f"play_table_sha256={actual_hashes['calvin_env/envs/play_table_env.py']}"
        )
        mismatches = {
            path: (actual_hashes[path], expected_hash)
            for path, expected_hash in EXPECTED_CALVIN_ENV_FILES.items()
            if actual_hashes[path] != expected_hash
        }
        if mismatches:
            raise RuntimeError(
                "These calvin_env files do not match the paper environment: "
                f"{mismatches}. Install/restore commit "
                f"{EXPECTED_CALVIN_ENV_COMMIT}."
            )
        if git_commit and git_commit != EXPECTED_CALVIN_ENV_COMMIT:
            logger.warning(
                "calvin_env git commit is %s rather than %s, but all "
                "execution-critical files match the paper revision.",
                git_commit,
                EXPECTED_CALVIN_ENV_COMMIT,
            )

    @staticmethod
    def _get_env_with_egl(abs_datasets_dir, observation_space, show_gui=False, **kwargs):
        """Instantiate CALVIN from the dataset config while forcing EGL.

        CALVIN's stock get_env() reloads merged_config.yaml internally, so
        changing a temporary OmegaConf object outside it has no effect.  Build
        the environment here to make the renderer choice explicit and
        auditable without modifying the dataset's config on disk.
        """
        merged_config = Path(abs_datasets_dir) / ".hydra" / "merged_config.yaml"
        render_conf = OmegaConf.load(merged_config)
        if not hasattr(render_conf, "env"):
            raise ValueError(f"Missing env config in {merged_config}")
        render_conf.env.use_egl = True

        if observation_space is not None:
            requested_cameras = {
                re.split("_", key, maxsplit=1)[1]
                for key in observation_space["rgb_obs"] + observation_space["depth_obs"]
            }
            for name in list(render_conf.cameras.keys()):
                if name not in requested_cameras:
                    del render_conf.cameras[name]

        if "scene" in kwargs:
            scene_cfg = OmegaConf.load(
                Path(calvin_env.__file__).parents[1]
                / "conf"
                / "scene"
                / f"{kwargs['scene']}.yaml"
            )
            render_conf.scene = scene_cfg

        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.initialize(".")
        return hydra.utils.instantiate(
            render_conf.env,
            show_gui=show_gui,
            use_vr=False,
            use_scene_info=True,
        )

    @staticmethod
    def set_egl_device(device):
        cuda_id = device.index if getattr(device, "type", None) == "cuda" else 0
        if cuda_id is None:
            cuda_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        try:
            egl_id = get_egl_device_id(cuda_id)
        except EglDeviceNotFoundError as exc:
            raise RuntimeError(
                "Could not map the CUDA device to an EGL device. Install the "
                "CALVIN/PyBullet EGL dependencies before running the paper "
                "evaluation; falling back to TinyRenderer changes observations."
            ) from exc
        requested = str(egl_id)
        current = os.environ.get("EGL_VISIBLE_DEVICES")
        if current is not None and current != requested:
            raise RuntimeError(
                f"EGL_VISIBLE_DEVICES={current} does not match CUDA device "
                f"mapping {cuda_id}->{requested}. Unset it before evaluation."
            )
        os.environ["EGL_VISIBLE_DEVICES"] = requested
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")

    def step(
        self, action_tensor: torch.Tensor
    ) -> Tuple[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], int, bool, Dict]:
        if self.relative_actions:
            action = action_tensor#.squeeze().cpu().detach().numpy()
            assert len(action) == 7
        else:
            if action_tensor.shape[-1] == 7:
                slice_ids = [3, 6]
            elif action_tensor.shape[-1] == 8:
                slice_ids = [3, 7]
            else:
                logger.error("actions are required to have length 8 (for euler angles) or 9 (for quaternions)")
                raise NotImplementedError
            action = np.split(action_tensor, slice_ids)
        o, r, d, i = self.env.step(action)

        obs = o # use raw observation
        self._last_obs = obs
        return obs, r, d, i

    def reset(
        self,
        reset_info: Dict[str, Any] = None,
        batch_idx: int = 0,
        seq_idx: int = 0,
        scene_obs: Any = None,
        robot_obs: Any = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        if reset_info is not None:
            obs = self.env.reset(
                robot_obs=reset_info["robot_obs"][batch_idx, seq_idx],
                scene_obs=reset_info["scene_obs"][batch_idx, seq_idx],
            )
        elif scene_obs is not None or robot_obs is not None:
            obs = self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
        else:
            obs = self.env.reset()

        self._last_obs = obs
        return obs # use raw observation

    def get_info(self):
        return self.env.get_info()

    def get_obs(self):
        obs = self.env.get_obs()
        # Some CALVIN versions return an empty observation before the first
        # reset, or expose the freshly-reset observation only as the return
        # value of reset().  Reuse the last valid observation in that case.
        if self._has_rgb_observation(obs):
            self._last_obs = obs
            return obs
        if self._has_rgb_observation(self._last_obs):
            return self._last_obs
        return obs # use raw observation

    @staticmethod
    def _has_rgb_observation(obs):
        return (
            isinstance(obs, dict)
            and isinstance(obs.get("rgb_obs"), dict)
            and bool(obs["rgb_obs"])
        )


    def action_space(self):
        return self.env.action_space


    def observation_space(self):
        return self.env.observation_space
