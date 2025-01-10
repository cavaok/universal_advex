#!/bin/bash
#SBATCH -N 1                  # Number of nodes
#SBATCH -n 1                  # Number of tasks
#SBATCH --mem=16G             # Memory per node
#SBATCH -t 6:00:00            # Time required
#SBATCH -p short              # Partition
#SBATCH -J adved              # Job name
#SBATCH -o advex_out.txt      # Standard output
#SBATCH -e advex_err.txt      # Standard error
#SBATCH --gres=gpu:L40S:1     # Request 1 L40S GPU

# Load Python environment
module load python/3.9.21/hgt2ae2

# Activate Environment
source /home/okcava/projects/universal_advex/.venv/bin/activate

# Run sweep scripts
python train_advex.py
