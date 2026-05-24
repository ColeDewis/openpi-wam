#!/bin/bash
#SBATCH --job-name=test_wam
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=%x-%j.out

module purge
unset PYTHONPATH
unset PYTHONHOME
module load python/3.11
module load cuda/12.2
module load cudnn/8.9.5.29
module load cmake/3.31.0
module load opencv/4.11.0
module load mujoco/3.3.0
module load gcc arrow/23.0.1

export OPENPI_REPO=/home/serg/serg/openpi-wam

echo "move venv"
# cp $OPENPI_REPO/venv311.tar $SLURM_TMPDIR/
cd $SLURM_TMPDIR
# tar -xf venv311.tar
#
# source .venv/bin/activate
source $OPENPI_REPO/.venv/bin/activate

echo "move data"
cp $OPENPI_REPO/hf_dataset.tar $SLURM_TMPDIR/
mkdir -p $SLURM_TMPDIR/huggingface
tar -xf hf_dataset.tar -C $SLURM_TMPDIR/huggingface

export HF_HOME=$SLURM_TMPDIR/huggingface
export HF_DATASETS_CACHE=$SLURM_TMPDIR/huggingface/datasets
export HF_DATASETS_OFFLINE=1

echo "starting"
python3 $OPENPI_REPO/scripts/train.py haptic_wam --exp-name=haptic_wam_test --overwrite
