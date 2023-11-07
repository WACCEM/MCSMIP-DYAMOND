import sys
import subprocess

if __name__ == "__main__":
    # Get PHASE from input ('Summer' or 'Winter')
    PHASE = sys.argv[1]

    root_indir = '/pscratch/sd/f/feng045/DYAMOND/OLR_Precipitation/'
    root_outdir = '/pscratch/sd/f/feng045/DYAMOND/OLR_Precipitation_combined/'

    # Activate E3SM environment to use NCO
    env_cmd = 'source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.sh'
    subprocess.run(env_cmd, shell=True)

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

    # Loop over list
    for run in runnames:
        in_dir = f'{root_indir}{PHASE}/{run}/'
        out_dir = f'{root_outdir}/'
        out_filename = f'{out_dir}olr_pcp_{PHASE}_{run}.nc'

        cmd = f'ncrcat -h {in_dir}*nc {out_filename}'
        print(cmd)
        subprocess.run(f'{cmd}', shell=True)
        # import pdb; pdb.set_trace()