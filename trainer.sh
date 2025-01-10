#!/bin/bash
#SBATCH -N 1                  # Number of nodes
#SBATCH -n 1                  # Number of tasks
#SBATCH --mem=16G             # Memory per node
#SBATCH -t 2:00:00            # Time required
#SBATCH -p short              # Partition
#SBATCH -J adved              # Job name
#SBATCH -o advex_out.txt      # Standard output
#SBATCH -e advex_err.txt      # Standard error
#SBATCH --gres=gpu:A100:1     # Request 1 A100 GPU

# Load Python environment
module load python/3.8.13/slu6jvw

# Activate Environment
source /home/okcava/projects/universal_advex/.venv/bin/activate

# Run sweep scripts
python train_advex.py