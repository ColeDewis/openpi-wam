import dataclasses
from pathlib import Path
import numpy as np
from termcolor import cprint

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
from scipy.spatial.transform import Rotation as R

# TODO: only libero works for now

# NOTE a bit lower than actual for safety
WAM_MIN_LIMITS = np.array([-2.5, -1.9, -2.6, -0.7, -4.5, -1.4, -2.9])
WAM_MAX_LIMITS = np.array([2.5, 1.9, 2.6, 2.9, 1.1, 1.4, 2.9])

WAM_FORCE_LIMIT  = 10
WAM_TORQUE_LIMIT =  0
 
KP_POS = 25   # N/m
KP_ROT = 1   # N·m/rad
 
def quat_to_euler(quat):
    """Quaternion [w, x, y, z] → Euler (roll, pitch, yaw) in radians."""
    quat_xyzw = np.roll(quat, -1, axis=-1)
    
    return R.from_quat(quat_xyzw).as_euler('xyz', degrees=False)

def delta_euler(rot_current, rot_next):
    """Minimal rotation delta between two quaternions, expressed as Euler xyz."""
    curr_xyzw = np.roll(rot_current, -1, axis=-1)
    next_xyzw = np.roll(rot_next, -1, axis=-1)
    
    r_delta = R.from_quat(curr_xyzw).inv() * R.from_quat(next_xyzw)
    return r_delta.as_euler('xyz', degrees=False)

def _euler_delta_to_rot_error(delta_euler_xyz: np.ndarray) -> np.ndarray:
    """
    Convert an incremental euler-XYZ delta (radians, EE frame) to a rotation
    error vector suitable for a proportional torque controller.
    """
    q = R.from_euler('xyz', delta_euler_xyz).as_quat()
    qx, qy, qz, qw = q
    return 2.0 * np.sign(qw) * np.array([qx, qy, qz])
 
 
def delta_to_wrench(delta: np.ndarray) -> np.ndarray:
    """
    Convert a single incremental delta [dx, dy, dz, droll, dpitch, dyaw, gripper]
    expressed in the EE frame into a wrench [fx, fy, fz, tx, ty, tz, gripper].
    """
    delta_pos        = delta[:3]
    delta_euler_xyz  = delta[3:6]
    gripper          = delta[6]
 
    force  = KP_POS * delta_pos
    torque = KP_ROT * _euler_delta_to_rot_error(delta_euler_xyz)
 
    force  = np.clip(force,  -WAM_FORCE_LIMIT,  WAM_FORCE_LIMIT)
    torque = np.clip(torque, -WAM_TORQUE_LIMIT, WAM_TORQUE_LIMIT)
 
    return np.concatenate([force, torque, [gripper]])
 


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
        print(f"trying to find norm_stats in {wam_assets.assets_dir} {wam_assets.asset_id}")
        
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
                "observation/gripper_position": np.zeros(1, dtype=np.float32),
                "prompt": "touch the green toy",
            }
        elif self.cfg_type == "libero":
            euler_rot = quat_to_euler(robot_state["follower_cart_rot"])
            example = {
                "observation/image": front_image,
                "observation/wrist_image": wrist_image,
                "observation/state": np.concatenate(
                    [robot_state["follower_cart_pos"], euler_rot, [robot_state["gripper_pos"]], [0]]
                ),
                "prompt": "touch the green thing",
            }
        else:
            raise Exception(f"Invalid cfg_type {self.cfg_type}")
 
        # [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action_chunk = self.policy.infer(example)["actions"]
 
        wrench_chunk = np.stack(
            [delta_to_wrench(d) for d in action_chunk]
        )
 
        if self.debug:
            for i, chunk in enumerate(wrench_chunk):
            # cprint(example, "cyan")
            # for i, (action) in enumerate(action_chunk):
                cprint(
                    f"step {i:02d} | "
                    # f"action: {np.round(action, 4)} | "
                    f"inc_pos: {np.round(chunk[:3], 4)} | "
                    f"inc_euler: {np.round(chunk[3:6], 4)} | "
                    f"gripper: {chunk[6]:.3f}",
                    "cyan",
                )
 
        return wrench_chunk


    def infer(self, obs):
        front_image = obs["front_image"]
        wrist_image = obs["wrist_image"]
        robot_state = obs["low_dim"]

        action_chunk = self._model_inference(front_image, wrist_image, robot_state)

        return action_chunk
