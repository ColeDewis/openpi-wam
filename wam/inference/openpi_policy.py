import dataclasses
import shutil
from pathlib import Path
import numpy as np
from termcolor import cprint

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config


# NOTE a bit lower than actual for safety
WAM_MIN_LIMITS = np.array([-2.5, -1.9, -2.6, -0.7, -4.5, -1.4, -2.9])
WAM_MAX_LIMITS = np.array([2.5, 1.9, 2.6, 2.9, 1.1, 1.4, 2.9])


class OpenPIPolicy:
    def __init__(self, checkpoint_path: str, model_config: str, debug: bool = False):
        self.debug = debug

        cprint(
            f"Loading policy with config: {model_config} from {checkpoint_path}...",
            "green",
        )
        self.checkpoint_path = checkpoint_path
        self.config = _config.get_config(model_config)
        wam_assets = _config.AssetsConfig(
            assets_dir="/home/coled/openpi/wam/config", asset_id="wam"
        )
        new_data_cfg = dataclasses.replace(self.config.data, assets=wam_assets)
        self.config = dataclasses.replace(self.config, data=new_data_cfg)
        checkpoint_dir = download.maybe_download(self.checkpoint_path)

        # openpi requires file to be present in the assets directory, so copy it there
        local_norm_path = Path("/home/coled/openpi/wam/config/wam/norm_stats.json")
        cached_assets_dir = Path(checkpoint_dir) / "assets" / "wam"
        cached_assets_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_norm_path, cached_assets_dir / "norm_stats.json")

        # Create a trained policy
        self.policy = policy_config.create_trained_policy(self.config, checkpoint_dir)

        # Do one initial inference on the policy to make sure it is fully loaded
        if "droid" in model_config:
            example = {
                "observation/exterior_image_1_left": np.random.randint(
                    256, size=(224, 224, 3), dtype=np.uint8
                ),
                "observation/wrist_image_left": np.random.randint(
                    256, size=(224, 224, 3), dtype=np.uint8
                ),
                "observation/joint_position": np.random.rand(7),
                "observation/gripper_position": np.random.rand(1),
                "prompt": "do something",
            }
            self.policy.infer(example)

    def _model_inference(self, front_image, wrist_image, robot_state):
        example = {
            "observation/exterior_image_1_left": front_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": robot_state,
            "observation/gripper_position": np.zeros(
                1, dtype=np.float32
            ),  # TODO figure out gripper stuff
            "prompt": "touch the red cup",
        }

        action_chunk = self.policy.infer(example)["actions"]

        # add joints to convert to absolute
        robot_state = np.concatenate(
            [robot_state, np.zeros(1)]
        )  # add dummy gripper state for now, fix later
        action_chunk = action_chunk + robot_state

        # clip by joint limits
        action_chunk[:, :7] = np.clip(
            action_chunk[:, :7], WAM_MIN_LIMITS, WAM_MAX_LIMITS
        )

        return action_chunk

    def infer(self, obs):
        front_image = obs["front_image"]
        wrist_image = obs["wrist_image"]
        jp_state = obs["follower_state"]["jp"]

        action_chunk = self._model_inference(front_image, wrist_image, jp_state)

        if self.debug:
            cprint("--- New chunk ---", "blue")
            for act in action_chunk:
                act = act[: self.DOF]
                cprint(f"Action from chunk: {np.round(act, 3)}", "cyan")

        return action_chunk
