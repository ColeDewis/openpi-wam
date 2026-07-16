#!/bin/bash
#SBATCH --job-name=pi0_job
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=%x-%j.out

source init_cc.sh

echo "move venv"
cd $SLURM_TMPDIR

echo "move data"
cp "${OPENPI_REPO}/hf_dataset.tar" $SLURM_TMPDIR/
tar -xf hf_dataset.tar -C $SLURM_TMPDIR

export HF_LEROBOT_HOME=$SLURM_TMPDIR
export HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# store base pi model in project dir instead of ~/.cache
export _OPENPI_DATA_HOME="${PROJECT_CACHE}/openpi"

echo "starting"
python3 $OPENPI_REPO/scripts/train.py haptic_wam_pi0 --exp-name=haptic_wam_pi0
