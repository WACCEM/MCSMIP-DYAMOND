"""
Prepare MCS tracking pixel data for visulization in VisIt.

Run this command to delete _FillValue attributes for cloudtracknumber, pcptracknumber 
in the output netCDF file for it to be plotted corrected in VisIt:
ncatted -a _FillValue,cloudtracknumber,d,l,0 <filename>

Example data for VisIt:
/global/cfs/cdirs/m1867/zfeng/E3SM/SCREAMv0/idl/mcs_1200x3600/data4visit
/global/cfs/cdirs/m1867/zfeng/dyamond-summer/obs/data4visit

Author: Zhe Feng, zhe.feng@pnnl.gov
History:
06/27/2023 - Written.
"""

import numpy as np
import glob, os, sys
import xarray as xr
import pandas as pd
import time
import yaml

def combine_data(mcs_files, outfilename, drop_varlist):

    # Read data files
    dsm = xr.open_mfdataset(mcs_files, drop_variables=drop_varlist, mask_and_scale=False).load()
    # import pdb; pdb.set_trace()

    # Roll lon=1800 to make the data start from 0~360
    dsm = dsm.roll(lon=1800, roll_coords=True)
    # Convert longitude coordinates from -180~180 to 0~360
    lon360 = dsm['lon'].data % 360
    dsm = dsm.assign_coords(lon=lon360)
    dsm['lon'].attrs['valid_min'] = 0.
    dsm['lon'].attrs['valid_max'] = 360.
    
    # Filter non-MCS tb, replace NaN with a large number
    tb_mcs = dsm['tb'].where(dsm['cloudtracknumber'] > 0, other=400)
    # Add to DataSet
    dsm['tb_mcs'] = tb_mcs

    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in dsm.data_vars}
    # Update time variable dtype as 'double' for better precision
    bt_dict = {'time': {'zlib':True, 'dtype':'float64'}}
    encoding.update(bt_dict)
    # Write to netcdf file
    dsm.to_netcdf(path=outfilename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'File saved: {outfilename}')

    return outfilename


if __name__=='__main__':

    # Get inputs
    config_file = sys.argv[1]
    run_name = sys.argv[2]
    start_date = sys.argv[3]
    end_date = sys.argv[4]

    # Get inputs from configuration file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)

    rootdir = config.get('indir')
    outdir = config.get('outdir')
    track_period = config.get('track_period')
    drop_varlist = config.get('drop_varlist')
    mcs_dir = f'{rootdir}{run_name}/mcstracking/{track_period}/'
    out_dir = f'{outdir}{run_name}/data4visit/'
    mcs_basename = 'mcstrack_'
    out_basename = 'mcstrack_'

    os.makedirs(out_dir, exist_ok=True)

    # Create a range of dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    mcs_dates = dates.strftime('%Y%m%d')
    # pw_dates = dates.strftime('%Y-%m-%d')

    outdate1 = dates[0].strftime('%Y%m%d')
    outdate2 = dates[-1].strftime('%Y%m%d')
    outfilename = f'{out_dir}{out_basename}{outdate1}_{outdate2}.nc'

    # Find MCS and PW files
    mcs_files = []
    # pw_files = []
    for ii in range(0, len(dates)):
        mcs_files.extend(sorted(glob.glob(f'{mcs_dir}{mcs_basename}{mcs_dates[ii]}*.nc')))
        # pw_files.extend(sorted(glob.glob(f'{pw_dir}{pw_basename}{pw_dates[ii]}*.nc')))
    nf_mcs = len(mcs_files)
    # nf_pw = len(pw_files)
    print(f'Number of MCS files: {nf_mcs}')
    # print(f'Number of PW files: {nf_pw}')

    # Process
    result = combine_data(mcs_files, outfilename, drop_varlist)

