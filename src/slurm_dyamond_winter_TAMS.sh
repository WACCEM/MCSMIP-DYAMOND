#!/bin/bash
#SBATCH -A m1867
#SBATCH -J WinTAMS
#SBATCH -t 00:20:00
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --output=log_Winter_TAMS_%A_%a.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov
#SBATCH --array=1-12

date
# Activate Python environment
source activate /global/common/software/m1867/python/pyflex

# cd /global/homes/f/feng045/program/mcsmip/dyamond/src/

# Takes a specified line ($SLURM_ARRAY_TASK_ID) from the task file
LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p tasks_mcsmip_Winter_TAMS.txt)
echo $LINE
# Run the line as a command
$LINE

date
