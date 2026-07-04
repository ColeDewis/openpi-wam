from huggingface_hub import HfApi

api = HfApi()
api.delete_repo(repo_id="Breakdancingbear/wam_teleop_dataset", repo_type="dataset")
