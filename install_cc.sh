# this script might not works first try. you might have to manually copy paste sections of the code until you install with 0 errors
module purge
module load python/3.11
module load cuda/12.2
module load cudnn/8.9.5.29    
module load cmake/3.31.0
module load opencv/4.11.0
module load mujoco/3.3.0 
module load gcc arrow/23.0.1

VENV_DIR="${HOME}/PI_ENV" 

virtualenv --no-download "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --no-index --upgrade pip

export PIP_ONLY_BINARY="mujoco,polars,polars-runtime-32"

git clone https://github.com/kvablack/dlimp.git
cd dlimp
git checkout ad72ce3a9b414db2185bc0b38461d4101a65477a
sed -i 's/tensorflow==2.15.0/tensorflow==2.17.0/g' setup.py
pip install -e .
cd ..

git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
sed -i '/rerun-sdk/d' pyproject.toml
sed -i 's/"opencv-python",//g' pyproject.toml
pip install -e .
cd ..

# This one below seems we can't win, half the packages want protobuf==5.29.6, other wants this, will have to test further.
pip install protobuf==4.25.8

pip install --no-index --no-deps -r wam/cc_conf/wheel_reqs.txt
# fix broken reqs
pip install --no-index "numpy==1.26.4" "jax==0.5.1" "jaxlib==0.5.1" "torch==2.7.1" "torchvision==0.22.1" "torchcodec==0.7.0"

pip install -r wam/cc_conf/pypi_reqs.txt

# you also need to import re in /home/<USER>/.local/lib/python3.11/site-packages/jaxtyping/__init__.py
sed -i 's|jax\[cuda12\]==0.5.3|jax[cuda12]>=0.5.1|g' pyproject.toml
sed -i 's|orbax-checkpoint==0.11.13|orbax-checkpoint==0.11.6|g' pyproject.toml
sed -i '/harvesters/d' pyproject.toml
pip install -e .

sed -i 's/^import warnings$/import warnings\nimport re/' \
    .venv/lib/python3.11/site-packages/jaxtyping/__init__.py
sed -i 's/timestamps = torch.stack(self.hf_dataset\["timestamp"\]).numpy()/timestamps = torch.stack(list(self.hf_dataset["timestamp"])).numpy()/' \
    lerobot/lerobot/common/datasets/lerobot_dataset.py
sed -i 's/torch.stack(self.hf_dataset\["\(.*\)"\])/torch.stack(list(self.hf_dataset["\1"]))/g' \
    lerobot/lerobot/common/datasets/lerobot_dataset.py
sed -i 's/key: torch.stack(self.hf_dataset.select(q_idx)\[key\])/key: torch.stack(list(self.hf_dataset.select(q_idx)[key]))/' \
    lerobot/lerobot/common/datasets/lerobot_dataset.py
pip install --no-index pytest

