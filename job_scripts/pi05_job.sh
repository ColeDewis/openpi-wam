#!/bin/bash
#SBATCH --job-name=pi05_job
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=%x-%j.out

source init_cc.sh
source "${OPENPI_REPO}/job_scripts/job_setup.sh"

echo "starting"
python3 $OPENPI_REPO/scripts/train.py haptic_wam_pi05 --exp-name=haptic_wam_pi05
