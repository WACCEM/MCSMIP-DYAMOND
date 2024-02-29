"""
Prepare ERA5 2D environmental files
- Combine two monthly files for each DYAMOND phase
- Rename lat, lon coordinate and inverse lat order
- Roll longitude to make the grid start from -180~+180, subset latitude to -59.75~+59.75
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

    # Subset latitude bounds
    lat_bounds = [-59.75, 59.75]

    # Make longitude grid from -180~+179.75
    lon180 = np.arange(-180, 180, 0.25)

    # Get first day of the month
    # this solves the issue for Winter where Jan is not included b/c the start_date is Jan 20
    first_day = pd.to_datetime(start_datetime).to_period('M').to_timestamp()
    # Make monthly datetime
    monthly_range = pd.date_range(start=first_day, end=end_datetime, inclusive='both', freq='MS')

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
    lon_attrs = ds['lon'].attrs
    # Reverse lat dimension so it goes from south to north
    ds = ds.reindex(lat=list(reversed(ds['lat'])))

    # Subset latitude
    ds = ds.sel(lat=slice(min(lat_bounds), max(lat_bounds)))
    # Roll lon=720 to make the data start from -180~+180
    ds = ds.roll(lon=720, roll_coords=True)
    # Replace longitude coordinate
    ds = ds.assign_coords(lon=lon180)
    ds['lon'].attrs = lon_attrs
    ds['lon'].attrs['valid_min'] = -180.
    ds['lon'].attrs['valid_max'] = 180.
    # Add latitude bounds
    ds['lat'].attrs['valid_min'] = min(lat_bounds)
    ds['lat'].attrs['valid_max'] = max(lat_bounds)

    # Rename variable
    dsout = ds.rename({varname_orig: varname})
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    print(f'Writing output ...')
    dsout.to_netcdf(path=out_filename, mode="w", format="NETCDF4", unlimited_dims='time', encoding=encoding)
    print(f"{out_filename}")