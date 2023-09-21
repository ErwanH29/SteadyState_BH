#!/bin/bash
#SBATCH --job-name=GRXres20
#SBATCH --export=NONE
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="ehochart@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --time=168:00:00
#SBATCH --partition=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --mem-per-cpu=10G

module load AMUSE/2021.3-CUDA-Miniconda3-4.11.0 

export CWD=$(pwd)
export DATE=$(date)

echo "[$SHELL] #### Starting Script"
echo "[$SHELL] ## current working directory: "$CWD
echo "[$SHELL] ## Run script"
python interface_resume.py
echo "[$SHELL] ## Script finished"
echo "[$SHELL] ## Job done "$DATE

echo "[$SHELL] ## Used $SLURM_NTASKS cores"
echo "[$SHELL] ## Used $SLURM_CPUS_ON_NODE processors/CPUS"
echo "[$SHELL] ## Used $SLURM_CPUS_PER_TASK processors/CPUS per task"

echo "[$SHELL] #### Finished Python run."

