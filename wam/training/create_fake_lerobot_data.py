import shutil
import os
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# --- Configuration ---
TASK_PROMPT = "synthetic test task"
SAVE_NAME = "test_user/fake_wam_dataset"  # Where the LeRobot dataset will be stored
NUM_EPISODES = 2
STEPS_PER_EPISODE = 50
FPS = 10


def main():
    # 1. Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / SAVE_NAME
    if output_path.exists():
        print(f"Cleaning up old dataset at {output_path}...")
        shutil.rmtree(output_path)

    # 2. Create LeRobot dataset, define features to store
    # Note: Shapes must exactly match what your training config expects
    dataset = LeRobotDataset.create(
        repo_id=SAVE_NAME,
        robot_type="wam",
        fps=FPS,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    print(f"Generating {NUM_EPISODES} fake episodes...")

    for ep_idx in range(NUM_EPISODES):
        for i in range(STEPS_PER_EPISODE):
            # Generate fake image data (random noise, uint8)
            # We add a colored block that moves slightly to simulate dynamic video
            fake_front = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
            fake_wrist = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)

            # Draw a moving square in the fake images
            pos = i % 256
            fake_front[pos : pos + 20, 50:70] = [255, 0, 0]  # Moving Red Square
            fake_wrist[100:120, pos : pos + 20] = [0, 255, 0]  # Moving Green Square

            # Generate fake state/actions (sine waves)
            # Shapes are (8,) as requested in your previous script
            t = i / float(STEPS_PER_EPISODE)
            fake_state = np.array(
                [np.sin(t * np.pi * 2 + j) for j in range(8)], dtype=np.float32
            )
            fake_action = np.array(
                [np.cos(t * np.pi * 2 + j) for j in range(8)], dtype=np.float32
            )

            dataset.add_frame(
                {
                    "image": fake_front,
                    "wrist_image": fake_wrist,
                    "state": fake_state,
                    "actions": fake_action,
                    "task": TASK_PROMPT,
                }
            )

        # Save the episode to disk
        dataset.save_episode()
        print(f"  Finished Episode {ep_idx + 1}/{NUM_EPISODES}")

    print(f"\n✅ Success! Fake dataset created at: {output_path}")
    print("You can now point your training script to this repo_id.")


if __name__ == "__main__":
    main()
