"""
Prepare ERA5 2D environmental files
- Combine two monthly files for each DYAMOND phase
- Rename lat, lon coordinate and inverse lat order
- Rename environmental variables to be consistent with unified DYAMOND data
"""
import numpy as np
import glob, sys, os
import xarray as xr
import pandas as pd
import yaml

if __name__ == "__main__":
    
    PHASE = sys.argv[1]
    runname = sys.argv[2]
    varname = sys.argv[3]

    # Config file contains input filenames for each model & environmental variable names
    config_file = f'/global/homes/f/feng045/program/mcsmip/dyamond/src/config_env_files_{PHASE}.yml'

    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/{PHASE}/{runname}/envs/'
    out_filename = f'{out_dir}{PHASE}_{runname}_{varname}.nc'
    os.makedirs(out_dir, exist_ok=True)

    # Start time for encoding 
    if PHASE == 'Summer':
        start_datetime = '2016-08-01 00:00:00'
        end_datetime = '2016-09-10 00:00:00'
    elif PHASE == 'Winter':
        start_datetime = '2020-01-20 00:00:00'
        end_datetime = '2020-03-01 00:00:00'

    # Make monthly datetime
    monthly_range = pd.date_range(start=start_datetime, end=end_datetime, freq='MS')

    # Get inputs from configuration file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)
    xcoord_name = config[runname]['xcoord_name']
    ycoord_name = config[runname]['ycoord_name']
    varname_orig = config[runname][varname]

    # Input filename
    in_dir = config[runname]['dirname']
    file_basename = config[runname]['file_basename']
    var_basename = config[runname][f'{varname}_basename']
    file_basename = f'{file_basename}{var_basename}'

    # Search files for each month
    in_files = []
    for imon in range(len(monthly_range)):
        # YearMonth sub-directory
        yearmon = monthly_range[imon].strftime('%Y%m')
        # Start date
        sdate = monthly_range[imon].strftime('%Y%m%d')
        # Search file
        in_files.extend(glob.glob(f'{in_dir}{yearmon}/{file_basename}{sdate}*nc'))
    print(f'ERA5 files: {in_files}')

    # Read input files, subset time period
    ds = xr.open_mfdataset(in_files, combine='by_coords').sel(time=slice(start_datetime, end_datetime))
    # Change time coordinate encoding
    ds['time'].encoding['units'] = f'hours since {start_datetime}'
    # Rename dimensions
    ds = ds.rename({xcoord_name:'lon', ycoord_name:'lat'})
    # Reverse lat dimension so it goes from south to north
    ds = ds.reindex(lat=list(reversed(ds['lat'])))

    # Rename variable
    dsout = ds.rename({varname_orig: varname})

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    print(f'Writing output ...')
    dsout.to_netcdf(path=out_filename, mode="w", format="NETCDF4", unlimited_dims='time', encoding=encoding)
    print(f"{out_filename}")
    
    # import pdb; pdb.set_trace()