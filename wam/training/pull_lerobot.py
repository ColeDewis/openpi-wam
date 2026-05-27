# python -m wam.training.pull_lerobot
import shutil
import tarfile
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main(
    repo_id: str = "Breakdancingbear/wam_teleop_dataset",
    local_dir: str = "./wam_teleop_dataset",
    tar_path: str = "./hf_dataset.tar",
):
    local_path = Path(local_dir)

    print(f"Pulling {repo_id} from HuggingFace Hub into {local_path} ...")
    dataset = LeRobotDataset(repo_id, root=local_path)
    dataset.pull_from_repo()
    print(f"Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}")

    print(f"Packing to {tar_path} ...")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(local_path, arcname=local_path.name)
    size_mb = Path(tar_path).stat().st_size / (1024 ** 2)

    print(f"Removing {local_path} ...")
    shutil.rmtree(local_path)

    print(f"Done. {tar_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
