#!/bin/bash
#SBATCH --job-name=new_params 
#SBATCH --time=30:00:00              
#SBATCH --array=0-4
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=40
#SBATCH --output=./bash_out/%x_%A_%a.out
#SBATCH --error=./bash_out/%x_%A_%a.err

#module load anaconda/2023a 

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

# Start Ray worker and keep the script running
echo "Starting Ray worker with temporary directory: $RAY_TMPDIR"
ray start --address='172.31.132.181:6379' --num-cpus=30 --num-gpus=1 --block 