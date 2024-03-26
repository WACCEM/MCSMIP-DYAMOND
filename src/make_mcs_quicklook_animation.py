"""
Make quicklook plots MCSMIP tracking.
"""
__author__ = "Zhe.Feng@pnnl.gov"

import sys, os
import yaml
import textwrap
import subprocess

if __name__ == "__main__":

    # Get PHASE from input ('Summer' or 'Winter')
    PHASE = sys.argv[1]

    # Submit slurm job
    # submit_job = True

    # Flag to call Python codes to make plots
    make_plots = True
    make_mp4 = True

    run_parallel = 1
    n_workers = 128

    code_dir = './'
    code_name = f'{code_dir}plot_subset_tbpf_mcs_tracks_1panel_mcsmip.py'
    fig_dir = '/global/cfs/cdirs/m1867/zfeng/MCSMIP/DYAMOND/'
    out_dir_mp4 = f'{fig_dir}{PHASE}/quicklooks/animations/'

    os.makedirs(out_dir_mp4, exist_ok=True)

    # DYAMOND phase start date
    if PHASE == 'Summer':
        # Domain extent (lonmin, lonmax, latmin, latmax)
        region = 'AFC'
        extent = [-5.0, 35.0, 0.0, 20.0]

        # region = 'WPAC'
        # extent = [125.0, 165.0, 10.0, 30.0]

        start_date = '2016-08-10T00'
        end_date = '2016-08-15T00'
    elif PHASE == 'Winter':
         # Domain extent (lonmin, lonmax, latmin, latmax)
        region = 'IO'
        extent = [55.0, 95.0, -20.0, 0.0]

        # region = 'AMZ'
        # extent = [-75, -35, -20.0, 0.0]

        start_date = '2020-02-01T00'
        end_date = '2020-02-06T00'

    # Tracker names
    # Trackers = ['PyFLEXTRKR']
    Trackers = [
        'PyFLEXTRKR',
        'MOAAP',
        'TOOCAN',
        'tobac',
        'TAMS',
        'simpleTrack',
    ]        

    # Model names
    if (PHASE == 'Summer'):
        # runnames = ['OBSv7', 'SCREAMv1']
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
        # runnames = ['OBSv7', 'SCREAMv1']
        runnames = [
            'ARPEGE',
            'GEOS',
            'GRIST',
            'ICON',
            'IFS',
            'MPAS',
            # 'NICAM',
            'OBS',
            'SAM',
            'SCREAM',
            'UM',
            'XSHiELD',
        ]
    else:
        print(f'Unknown PHASE: {PHASE}') 

    # Video
    framerate = 2
    vfscale = '1200:-1'

    # Loop over Trackers
    for tracker in Trackers:
        # Loop over list
        for rname in runnames:
            phase = PHASE.lower()
            quicklook_dir = f'{fig_dir}{PHASE}/quicklooks/{tracker}/{rname}/'
            figbasename = f'{PHASE}_{rname}_{region}_'
            
            cmd = f"python {code_name} -s {start_date} -e {end_date} " + \
                    f" --extent {extent[0]} {extent[1]} {extent[2]} {extent[3]} --output {quicklook_dir} --figbasename {figbasename} " + \
                    f" --phase {PHASE} --runname {rname} --tracker {tracker} --subset 1 -p {run_parallel} --n_workers {n_workers}"
            if make_plots == True:
                print(cmd)
                subprocess.run(cmd, shell=True)
            
            # Make animation
            video_filename = f'{out_dir_mp4}{PHASE}_{rname}_{tracker}_{region}.mp4'
            # Make ffmpeg command
            cmd = f"ffmpeg -framerate {framerate} -pattern_type glob -i '{quicklook_dir}{figbasename}*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p -vf scale={vfscale} -y {video_filename}"
            # print(cmd)
            # import pdb; pdb.set_trace()
            if make_mp4 == True:
                print(video_filename)
                subprocess.run(cmd, shell=True)
        #     task_file.write(f"{cmd}\n")
        #     ntasks += 1

        # task_file.close()
        # print(task_filename)