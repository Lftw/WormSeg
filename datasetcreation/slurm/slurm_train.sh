#!/bin/bash
#SBATCH --job-name=segformer # Job name
#SBATCH --account=def-mzhen # Your account
#SBATCH --time=00-4:00 # Time limit (DD-HH:MM)
#SBATCH --cpus-per-task=6 # CPUs per task (you can increase if you want more DataLoader workers)
#SBATCH --gres=gpu:1 # Number of GPUs per node
#SBATCH --ntasks-per-node=1 # <-- launch 2 tasks (ranks) on this node
#SBATCH --mem=64G # Memory per CPU
#SBATCH --output=logs/slurm/wormseg-x-%j.out


module load python
module load StdEnv/2023
module load gcc cuda arrow
module load opencv
module load rust
module load scipy-stack/2024a
module load blender
module load cmake
module load python-build-bundle/2023b
module list

cd projects/def-mzhen/alexw/Projects/WormSeg

source worms_env/bin/activate
cd WormSeg/datasetcreation/slurm
cd .. # to move back to your root repository
srun python3 train.py 
