#!/bin/bash
# Runs all MCS diagnostics for a single source and tracker
# Most of the codes use parallel processing or requires large memory, 
# this script must be run in an exclusive compute node
# Example:
# ./run_alljobs_1source.sh Summer SCREAMv1 PyFLEXTRKR

# Get PHASE from input ('Summer' or 'Winter')
PHASE=$1
# Get runname from input ('OBS', 'SCREAM')
runname=$2
# Name of the tracker (e.g., 'PyFLEXTRKR', 'MOAAP')
tracker=$3

# Check if exactly three arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 PHASE runname tracker"
    exit 1
fi

# Environment variable name
env_varname='intqv'

# Activate Python environment
source activate /global/common/software/m1867/python/pyflex

# Start/end dates
if [[ $PHASE == 'Summer' ]]
then
    start_date='2016-08-10T00'
    end_date='2016-09-10T00'
elif [[ $PHASE == 'Winter' ]]
then
    start_date='2020-02-01T00'
    end_date='2020-03-01T00'
else
    echo Unknown PHASE: $PHASE
fi

# Config file
config_dir=/global/homes/f/feng045/program/pyflex_config/config/
config_basename=config_dyamond_
config_file=${config_dir}${config_basename}${PHASE}_${runname}.yml

# Unify Tb/OLR-precipitation data
python unify_dyamond_olr_pcp_files.py ${config_file} ${PHASE} ${runname}

# Combine hourly Tb/OLR-precipitation files to a single file using ncks
# This step may need to be run separately on the command line
# Activate E3SM unified environment
source deactivate
source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.sh
echo 'Combining unified Tb/OLR-precipitation to a single file ...'
ncrcat -h olr_pcp_*.nc olr_pcp_${PHASE}_${runname}.nc
which ncks
source deactivate
source activate /global/common/software/m1867/python/pyflex

# MCS
if [[ ${tracker} == 'PyFLEXTRKR' ]]
then
    python make_mcs_maskfile_singlefile.py ${PHASE} ${runname}
else
    python unify_mask_files.py ${PHASE} ${runname} ${tracker}
fi
python make_mcs_stats_from_maskfile.py ${config_file} ${tracker}
python avg_global_rain_timeseries.py ${PHASE} ${runname}
python calc_tbpf_mcs_rainmap_mcsmip.py ${config_file} ${tracker} ${start_date} ${end_date}
python calc_tb_rainrate_pdf_byregion.py ${PHASE} ${runname} ${tracker}

# Environments
python unify_env_files.py ${PHASE} ${runname} ${env_varname}
python extract_mcs_2d_env.py ${PHASE} ${runname} ${tracker} ${env_varname}
python avg_mcs_track_env_space.py ${PHASE} ${runname} ${tracker} ${env_varname}
python regrid_envs2era5.py ${PHASE} ${runname} ${env_varname}
python regrid_tbpcp2era5.py ${PHASE} ${runname}
python regrid_mcsmask2era5.py ${PHASE} ${runname} ${tracker}
python calc_mcs_pcp_envs_pairs.py ${PHASE} ${runname} ${tracker} ${env_varname} ${start_date} ${end_date}