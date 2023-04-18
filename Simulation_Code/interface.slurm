#!/bin/bash
#SBATCH --job-name=GRX_Code
#SBATCH --export=NONE
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="ehochart@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --time=120:00:00
#SBATCH --partition=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --mem-per-cpu=10G

# load modules (assuming you start from the default environment)
# module load pandas/0.24.0-foss-2018a-Python-3.6.4 
# module load pandas/0.24.0-foss-2019a-Python-3.
#module load AMUSE/2022.3-Miniconda3-4.11.0
module load AMUSE/2021.3-CUDA-Miniconda3-4.11.0 
# source $HOME/python_test_venv/bin/activate

export CWD=$(pwd)
export DATE=$(date)
echo "[$SHELL] #### Starting Script"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
# get the current working directory
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"

python interface.py

echo "[$SHELL] ## Script finished"
echo "[$SHELL] ## Job done "$DATE
echo "[$SHELL] ## Used $SLURM_NTASKS cores"
echo "[$SHELL] ## Used $SLURM_CPUS_ON_NODE processors/CPUS"
echo "[$SHELL] ## Used $SLURM_CPUS_PER_TASK processors/CPUS per task"

echo "[$SHELL] #### Finished Python test. Have a nice day"

