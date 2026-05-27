# python -m wam.training.pull_lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("Breakdancingbear/wam_teleop_dataset")
print(f"Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}")
