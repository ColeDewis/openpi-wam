from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np

def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)


example = _random_observation_droid()
print(example.keys())

# Run inference on a dummy example.
# example = {
    # "observation/exterior_image_1_left": ...,
    # "observation/wrist_image_left": ...,
    # ...
    # "prompt": "pick up the fork"
# }config.model.fake_obs().to_dict()
preds = policy.infer(example)
action = preds["actions"]
time = preds["policy_timing"]["infer_ms"]

print(time)

preds = policy.infer(example)
action = preds["actions"]
time = preds["policy_timing"]["infer_ms"]

print(action, time)