#!/bin/bash
# Runs prepare data for VisIt for all models

# Get PHASE from input ('Summer' or 'Winter')
PHASE=$1
# Name of the tracker (e.g., 'PyFLEXTRKR', 'MOAAP')
tracker=$2

# Activate Python environment
source activate /global/common/software/m1867/python/pyflex

# Python code
code_dir='/global/homes/f/feng045/program/mcsmip/dyamond/src/'
code_name=${code_dir}'prepare_mcs_pwv_data4visit_mcsmip.py'

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
    # runnames=('SCREAMv1' 'MPAS')
    runnames=(
        'ARPEGE'
        'GEOS'
        'GRIST'
        'ICON'
        'IFS'
        'MPAS'
        'NICAM'
        'OBS'
        'OBSv7'
        'SAM'
        'SCREAM'
        'SCREAMv1'
        'UM'
        'XSHiELD'
    )
else
    echo Unknown PHASE: $PHASE
fi

# Loop over each run
for run in "${runnames[@]}"; do
    echo ${run}
    python ${code_name} ${PHASE} ${tracker} ${run}
done