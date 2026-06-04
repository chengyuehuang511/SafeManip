#!/bin/bash
#SBATCH --job-name=compile_amago
#SBATCH --output=compile_%j.log
#SBATCH --error=compile_%j.err
#SBATCH --time=90:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G

# 1. Clear existing modules and load the exact compilation stack
module purge
module load miniforge/25.3.1-py3.12
module load gcc/12.4.0
module load cuda/12.8.1

# 2. Initialize mamba inside the non-interactive batch shell
conda activate /your/environment/here

# 3. Route uv cache to the compute node's local storage to prevent file locking issues
export UV_CACHE_DIR=/tmp/$USER/uv_cache_$SLURM_JOB_ID
export UV_LOCK_TIMEOUT=72000
export PIP_CACHE_DIR=/your/pip/cache/here

# 4. Navigate to your project directory
cd /your/project/directory

# 5. Clear any lingering stale locks just in case
rm -f /found/uv/log/and/put/the/absolute/path

export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export ATTN_VERS="2.7.4.post1"

# 6. Run the installation
echo "Starting compilation..."
uv pip install flash-attn==$ATTN_VERS --no-build-isolation
echo "Finished compilation."