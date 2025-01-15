"""
Make MCS mask swath plots for MCSMIP.
"""
__author__ = "Zhe.Feng@pnnl.gov"

import sys, os
import subprocess

if __name__ == "__main__":

    # Get PHASE from input ('Summer' or 'Winter')
    PHASE = sys.argv[1]

    # Flag to call Python codes to make plots
    make_plots = True

    code_dir = './'
    code_name = f'{code_dir}plot_subset_mcs_maskswath_mcsmip.py'
    fig_dir = f'/global/cfs/cdirs/m1867/zfeng/MCSMIP/DYAMOND/{PHASE}/quicklooks/swath_masks/'

    os.makedirs(fig_dir, exist_ok=True)

    # DYAMOND phase start date
    if PHASE == 'Summer':
        # Domain extent (lonmin, lonmax, latmin, latmax)
        region = 'WPAC'
        extent = [125.0, 165.0, 10.0, 30.0]
        start_date = '2016-08-10T00'
        end_date = '2016-08-11T00'

        # region = 'AFC'
        # extent = [-5.0, 35.0, 0.0, 20.0]
        # start_date = '2016-08-11T10'
        # end_date = '2016-08-12T11'
        
    elif PHASE == 'Winter':
        # Domain extent (lonmin, lonmax, latmin, latmax)
        region = 'IO'
        extent = [55.0, 95.0, -20.0, 0.0]
        start_date = '2020-02-01T00'
        end_date = '2020-02-02T00'

        # region = 'AMZ'
        # extent = [-75, -35, -20.0, 0.0]
        # start_date = '2020-02-03T12'
        # end_date = '2020-02-04T13'


    # Tracker names
    Trackers = ['ATRACKCS']
    # Trackers = [
    #     'PyFLEXTRKR',
    #     'MOAAP',
    #     'TOOCAN',
    #     'tobac',
    #     'TAMS',
    #     'simpleTrack',
    #     'DL',
    #     'KFyAO',
    #     'TIMPS',
    #     'ATRACKCS',
    # ]

    # Model names
    if (PHASE == 'Summer'):
        # runnames = ['OBS']
        runnames = [
            # 'ARPEGE',
            # 'FV3',
            # 'IFS',
            # 'MPAS',
            # 'NICAM',
            'OBS',
            # 'OBSv7',
            # 'SAM',
            # 'UM',
            # 'SCREAMv1',
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
            'OBSv7',
            'SAM',
            'SCREAM',
            'SCREAMv1',
            'UM',
            'XSHiELD',
        ]
    else:
        print(f'Unknown PHASE: {PHASE}') 

    # Loop over Trackers
    for tracker in Trackers:
        # Loop over runnames
        for rname in runnames:
            # phase = PHASE.lower()
            # quicklook_dir = f'{fig_dir}{PHASE}/quicklooks/{tracker}/{rname}/'
            figbasename = f'{PHASE}_{rname}_{region}_'
            
            cmd = f"python {code_name} -s {start_date} -e {end_date} " + \
                    f" --extent {extent[0]} {extent[1]} {extent[2]} {extent[3]} --output {fig_dir} " + \
                    f" --phase {PHASE} --runname {rname} --tracker {tracker} --region {region} --subset 1"
            if make_plots == True:
                print(cmd)
                subprocess.run(cmd, shell=True)
