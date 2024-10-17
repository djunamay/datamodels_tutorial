#!/bin/bash
#SBATCH --job-name=headnode 
#SBATCH --time=60:00:00              
#SBATCH --exclusive
#SBATCH --output=./bash_out/%x_%A_%a.out
#SBATCH --error=./bash_out/%x_%A_%a.err

# Create unique Ray temporary directory with appropriate permissions
RAY_TMPDIR="/tmp/ray_$(date +%s)_$RANDOM"
mkdir -p "$RAY_TMPDIR"
chown -R :TsaiMadry "$RAY_TMPDIR"
chmod -R 770 "$RAY_TMPDIR"
export RAY_TMPDIR="$RAY_TMPDIR"

export RAY_ROTATION_MAX_BYTES=1024
export RAY_ROTATION_BACKUP_COUNT=1

# Function to clean up Ray resources
cleanup() {
    echo "Stopping Ray..."
    ray stop
    echo "Removing temporary Ray directory..."
    rm -rf "$RAY_TMPDIR"
}

# Ensure that cleanup is called on script exit, regardless of success or failure
trap cleanup EXIT

# Start Ray head
ray start --head --num-cpus=40 --num-gpus=0 --temp-dir="$RAY_TMPDIR"

# Launch training
python train_cifar_with_ray.py --config-file "./configs/updated_train_with_ray.yaml"
