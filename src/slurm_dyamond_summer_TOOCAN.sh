#!/bin/bash
#SBATCH -A m1867
#SBATCH -J Summer_TOOCAN
#SBATCH -t 03:00:00
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --output=log_Summer_TOOCAN_%A_%a.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov
#SBATCH --array=1-8

date
# Activate Python environment
source activate /global/common/software/m1867/python/pyflex

# cd /global/homes/f/feng045/program/mcsmip/dyamond/src/

# Takes a specified line ($SLURM_ARRAY_TASK_ID) from the task file
LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p tasks_mcsmip_Summer_TOOCAN.txt)
echo $LINE
# Run the line as a command
$LINE

date
