#!/bin/bash
# Runs make MCS maskfile or stats file for pecified models

# Get PHASE from input ('Summer' or 'Winter')
PHASE=$1
# Name of the tracker (e.g., 'PyFLEXTRKR', 'MOAAP')
tracker=$2

# Activate Python environment
source activate /global/common/software/m1867/python/pyflex

# Python code
code_dir='/global/homes/f/feng045/program/dyamond/src/'
code_name=${code_dir}'unify_mask_files.py'
# code_name=${code_dir}'make_mcs_maskfile_singlefile.py'
# code_name=${code_dir}'make_mcs_stats_from_maskfile.py'

# Model names
if [[ $PHASE == 'Summer' ]]
then
    # runnames=('MPAS')
    runnames=(
        'ARPEGE'
        'FV3'
        'IFS'
        'MPAS'
        'NICAM'
        'OBS'
        'SAM'
        'UM'
    )
elif [[ $PHASE == 'Winter' ]]
then
    # runnames=('SCREAM' 'XSHiELD')
    runnames=(
        'ARPEGE'
        'GEOS'
        'GRIST'
        'ICON'
        'IFS'
        'MPAS'
        'NICAM'
        'OBS'
        'SAM'
        'SCREAM'
        'UM'
        'XSHiELD'
    )
else
    echo Unknown PHASE: $PHASE
fi

# Loop over each run
for run in "${runnames[@]}"; do
    echo ${run}
    python ${code_name} ${PHASE} ${run} ${tracker}
done