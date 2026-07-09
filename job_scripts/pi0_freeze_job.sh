#!/bin/bash
#SBATCH --job-name=test_wam
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=%x-%j.out

module purge
module load python/3.11
module load cuda/12.2
module load cudnn/8.9.5.29
module load cmake/3.31.0
module load opencv/4.11.0
module load mujoco/3.3.0
module load gcc
module load arrow/23.0.1

export OPENPI_REPO=/home/serg/projects/openpi-wam
# export OPENPI_REPO=/project/def-jag/serg/openpi-wam

echo "move venv"
cd $SLURM_TMPDIR

source $OPENPI_REPO/.venv/bin/activate

echo "move data"
cp $OPENPI_REPO/hf_dataset.tar $SLURM_TMPDIR/
mkdir -p $SLURM_TMPDIR/huggingface
tar -xf hf_dataset.tar -C $SLURM_TMPDIR

export HF_LEROBOT_HOME=$SLURM_TMPDIR
export HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

echo "starting"
python3 $OPENPI_REPO/scripts/train.py haptic_wam_pi0_freeze --exp-name=haptic_wam_pi0_freeze
