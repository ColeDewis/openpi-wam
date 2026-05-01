module load python/3.11.5
module load mujoco/3.3.0
module load arrow/21.0.0

virtualenv --no-download .venv
source .venv/bin/activate

pip install -r wam/cc_conf/wheel_reqs.txt
pip install -r wam/cc_conf/pypi_reqs.txt
