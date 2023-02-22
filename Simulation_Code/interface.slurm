#!/bin/bash
#SBATCH --job-name=SteadyStateBH_Orbits_ALICE_Code_1020
#SBATCH --export=NONE
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="ehochart@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --time=120:00:00
#SBATCH --partition=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=14

module load AMUSE/2022.7-Miniconda3-4.11.0

echo "[$SHELL] #### Starting Script"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"

# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
echo "[$SHELL] ## Job start "$DATE
python3 interface.py
echo "[$SHELL] ## Script finished"
echo "[$SHELL] ## Job done "$DATE
echo "[$SHELL] #### Finished Python test. Have a nice day"
