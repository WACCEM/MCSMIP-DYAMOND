"""
Make task list and slurm scripts for MCSMIP statistics.
"""
__author__ = "Zhe.Feng@pnnl.gov"

import sys, os
import numpy as np
import xarray as xr
import yaml
import textwrap
import subprocess

if __name__ == "__main__":

    # Get PHASE from input ('Summer' or 'Winter')
    PHASE = sys.argv[1]
    # Name of the tracker (e.g., 'PyFLEXTRKR', 'MOAAP')
    tracker = sys.argv[2]

    # Submit slurm job
    submit_job = True

    code_dir = '/global/homes/f/feng045/program/mcsmip/dyamond/src/'
    # slurm_dir = code_dir
    # code_name = f'{code_dir}unify_dyamond_olr_pcp_files.py'
    # code_name = f'{code_dir}unify_mask_files.py'
    # code_name = f'{code_dir}make_mcs_maskfile_singlefile.py'
    code_name = f'{code_dir}calc_tbpf_mcs_rainmap_mcsmip.py'
    # code_name = f'{code_dir}make_mcs_stats_from_maskfile.py'
    config_dir = '/global/homes/f/feng045/program/pyflex_config/config/'
    config_basename = f'config_dyamond_'
    slurm_basename = f'slurm_dyamond_'

    # Set wallclock_time based on which code is used
    if 'unify_dyamond_olr_pcp_files' in code_name:
        wallclock_time = '00:05:00'
    elif 'unify_mask_files' in code_name:
        wallclock_time = '00:10:00'
    elif 'make_mcs_maskfile_singlefile' in code_name:
        wallclock_time = '00:15:00'
    elif 'calc_tbpf_mcs_rainmap_mcsmip' in code_name:
        wallclock_time = '00:10:00'
    elif 'make_mcs_stats_from_maskfile' in code_name:
        wallclock_time = '00:20:00'

    # DYAMOND phase start date
    if PHASE == 'Summer':
        # start_date = '2016-08-01T00'
        start_date = '2016-08-10T00'
        end_date = '2016-09-10T00'
    elif PHASE == 'Winter':
        # start_date = '2020-01-20T00'
        start_date = '2020-02-01T00'
        end_date = '2020-03-01T00'

    # Model names
    if (PHASE == 'Summer'):
        # runnames = ['MPAS']
        runnames = [
            'ARPEGE',
            'FV3',
            'IFS',
            'MPAS',
            'NICAM',
            'OBS',
            'SAM',
            'UM',
        ]
    elif (PHASE == 'Winter'):
        # runnames=['SCREAM', 'XSHiELD']
        runnames = [
            'ARPEGE',
            'GEOS',
            'GRIST',
            'ICON',
            'IFS',
            'MPAS',
            'NICAM',
            'OBS',
            'SAM',
            'SCREAM',
            'UM',
            'XSHiELD',
        ]
    else:
        print(f'Unknown PHASE: {PHASE}') 

    # Create the list of job tasks needed by SLURM...
    task_filename = f'tasks_mcsmip_{PHASE}_{tracker}.txt'
    task_file = open(task_filename, "w")
    ntasks = 0
            
    # Loop over list
    for run in runnames:
        phase = PHASE.lower()
        config_file = f'{config_dir}{config_basename}{phase}_{run}.yml'
        if os.path.isfile(config_file) != True:
            print(f'ERROR: config file does NOT exist: {config_file}')
            sys.exit(f'Code will exist now.')
        if 'unify_dyamond_olr_pcp_files' in code_name:
            cmd = f'python {code_name} {config_file}'
        elif 'unify_mask_files' in code_name:
            cmd = f'python {code_name} {PHASE} {run} {tracker}'
        else:
            cmd = f'python {code_name} {config_file} {tracker} {start_date} {end_date}'
        task_file.write(f"{cmd}\n")
        ntasks += 1

    task_file.close()
    print(task_filename)
    # import pdb; pdb.set_trace()


    # Create a SLURM submission script for the above task list...
    slurm_filename = f'{slurm_basename}{phase}_{tracker}.sh'
    slurm_file = open(slurm_filename, "w")
    text = f"""\
        #!/bin/bash
        #SBATCH -A m1867
        #SBATCH -J {PHASE}{tracker}
        #SBATCH -t {wallclock_time}
        #SBATCH -q regular
        #SBATCH -C cpu
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=128
        #SBATCH --exclusive
        #SBATCH --output=log_{PHASE}_{tracker}_%A_%a.log
        #SBATCH --mail-type=END
        #SBATCH --mail-user=zhe.feng@pnnl.gov
        #SBATCH --array=1-{ntasks}

        date
        # Activate Python environment
        source activate /global/common/software/m1867/python/pyflex

        # cd {code_dir}

        # Takes a specified line ($SLURM_ARRAY_TASK_ID) from the task file
        LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p {task_filename})
        echo $LINE
        # Run the line as a command
        $LINE

        date
    """
    slurm_file.writelines(textwrap.dedent(text))
    slurm_file.close()
    print(slurm_filename)

    # Run command
    if submit_job == True:
        cmd = f'sbatch --array=1-{ntasks} {slurm_filename}'
        print(cmd)
        subprocess.run(f'{cmd}', shell=True)

        # import pdb; pdb.set_trace()