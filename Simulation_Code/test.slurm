#!/bin/bash
#SBATCH --job-name=Testing_SteadyStateBH
#SBATCH --export=NONE
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

module load AMUSE/2022.7-Miniconda3-4.11.0

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
