#!/bin/bash
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 1                   # Number of tasks
#SBATCH --mem=16G              # Memory per node
#SBATCH -t 48:00:00           # Time required - increased to 48 hours for all sweeps
#SBATCH -p short              # Partition
#SBATCH -J auto_sweeps        # Job name
#SBATCH -o sweep_out.txt      # Standard output
#SBATCH -e sweep_err.txt      # Standard error
#SBATCH --gres=gpu:A100:1     # Request 1 A100 GPU

# Load Python environment
module load python/3.8.13/slu6jvw

# Activate Environment
source /home/okcava/projects/universal_advex/.venv/bin/activate

# Run sweep scripts
python auto_sweep_512_elu.py
python auto_sweep_512_sigmoid.py
python auto_sweep_256_elu.py
python auto_sweep_256_sigmoid.py
python auto_sweep_128_elu.py
python auto_sweep_128_sigmoid.py
python auto_sweep_64_elu.py
python auto_sweep_64_sigmoid.py