#!/bin/bash

cd $SLURM_TMPDIR
echo "move data"
cp "${OPENPI_REPO}/hf_dataset.tar" $SLURM_TMPDIR/
tar -xf hf_dataset.tar -C $SLURM_TMPDIR

export HF_LEROBOT_HOME=$SLURM_TMPDIR
export HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# store base pi models in project dir instead of ~/.cache
export OPENPI_DATA_HOME="${PROJECT_CACHE}/openpi"
