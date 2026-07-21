#!/bin/bash
module load python/3.11
module load cuda/12.2
module load cudnn/8.9.5.29    
module load cmake/3.31.0
module load opencv/4.11.0
module load mujoco/3.3.0 
module load gcc arrow/23.0.1

# ==============================================================================
# USER CONFIGURATION (Edit these paths to match your environment)
# ==============================================================================

export OPENPI_REPO=/home/serg/projects/openpi-wam
# export OPENPI_REPO=/project/def-jag/serg/openpi-wam
export OPENPI_ASSETS_DIR="${OPENPI_REPO}/assets/"
export OPENPI_CHECKPOINTS_DIR="${HOME}/scratch/openpi-wam/checkpoints/"

# ==============================================================================
# END USER CONFIGURATION
# ==============================================================================

export PROJECT_CACHE="${OPENPI_REPO}/.cache"
mkdir -p "${PROJECT_CACHE}"

source "${HOME}/PI_ENV/bin/activate"
# source /project/def-jag/serg/openpi-wam/.venv/bin/activate
