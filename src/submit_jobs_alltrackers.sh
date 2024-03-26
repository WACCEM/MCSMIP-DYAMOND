#!/bin/bash
# Submits slurm jobs to get MCSMIP statistics for all trackers

# Get season from input
season=$1

# Declare an array containing tracker names
declare -a list=("MOAAP" "PyFLEXTRKR" "simpleTrack" "TAMS" "tobac" "TOOCAN")
# declare -a list=("MOAAP" "simpleTrack" "TAMS" "tobac" "TOOCAN")

# Loop through the list
for ii in "${list[@]}"; do
    # echo $1 ${ii}
   python make_mcs_stats_joblib.py $1 ${ii}
done