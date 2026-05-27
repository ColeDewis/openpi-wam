import dataclasses
import shutil
from pathlib import Path
import numpy as np
from termcolor import cprint

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints


# NOTE a bit lower than actual for safety
WAM_MIN_LIMITS = np.array([-2.5, -1.9, -2.6, -0.7, -4.5, -1.4, -2.9])
WAM_MAX_LIMITS = np.array([2.5, 1.9, 2.6, 2.9, 1.1, 1.4, 2.9])


class OpenPIPolicy:
    def __init__(self, checkpoint_path: str, model_config: str, cfg_type: str = "libero", debug: bool = False, dof: int=7):
        self.debug = debug

        cprint(
            f"Loading {cfg_type} policy with config: {model_config} from {checkpoint_path}...",
            "green",
        )

        if cfg_type not in ["libero", "droid"]:
            cprint(
                f"cfg_type: {cfg_type} must be one of libero, droid",
                "red",
            )
            return

        self.DOF = dof
        self.checkpoint_path = checkpoint_path
        self.config = _config.get_config(model_config)
        wam_assets = _config.AssetsConfig(
            assets_dir="/home/serg/projects/openpi-wam/assets/haptic_wam/Breakdancingbear", asset_id="wam_teleop_dataset"
        )
        new_data_cfg = dataclasses.replace(self.config.data, assets=wam_assets)
        self.config = dataclasses.replace(self.config, data=new_data_cfg)
        checkpoint_dir = download.maybe_download(self.checkpoint_path)

        norm_stats = _checkpoints.load_norm_stats(wam_assets.assets_dir, wam_assets.asset_id)
        
        # Create a trained policy
        self.policy = policy_config.create_trained_policy(self.config, checkpoint_dir, norm_stats=norm_stats)

        self.cfg_type = cfg_type

        # Do one initial inference on the policy to make sure it is fully loaded
        if cfg_type == "droid":
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
        elif cfg_type == "libero":
            example = {
                "observation/image": np.random.randint(
                    256, size=(224, 224, 3), dtype=np.uint8
                ),
                "observation/wrist_image": np.random.randint(
                    256, size=(224, 224, 3), dtype=np.uint8
                ),
                "observation/state": np.random.rand(8),
                "prompt": "do something",
            }
            self.policy.infer(example)


    def _model_inference(self, front_image, wrist_image, robot_state):
        if self.cfg_type == "droid":
            example = {
                "observation/exterior_image_1_left": front_image,
                "observation/wrist_image_left": wrist_image,
                "observation/joint_position": robot_state["jp"],
                "observation/gripper_position": np.zeros(
                    1, dtype=np.float32
                ),  # TODO figure out gripper stuff
                "prompt": "touch the green toy",
            }
        elif self.cfg_type == "libero":
            example = {
                "observation/image": front_image,
                "observation/wrist_image": wrist_image,
                "observation/state": ( np.concatenate([robot_state["jp"], [robot_state["gripper"]]])),
                "observation/gripper_position": np.zeros(
                    1, dtype=np.float32
                ),  # TODO figure out gripper stuff
                "prompt": "touch the gren toy",
            }
        else:
            raise Exception(f"invalid cfg_type {self.cfg_type}")

        action_chunk = self.policy.infer(example)["actions"]

        # clip by joint limits
        action_chunk[:, :7] = np.clip(
            action_chunk[:, :7], WAM_MIN_LIMITS, WAM_MAX_LIMITS
        )

        return action_chunk

    def infer(self, obs):
        front_image = obs["front_image"]
        wrist_image = obs["wrist_image"]
        robot_state = obs["follower_state"]

        action_chunk = self._model_inference(front_image, wrist_image, robot_state)

        if self.debug:
            cprint("--- New chunk ---", "blue")
            for act in action_chunk:
                cprint(f"Action from chunk: {np.round(act, 3)}", "cyan")

        return action_chunk
