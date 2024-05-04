"""
Unify DYAMOND environment files:
- Harmonize time coordinate to CF-complient (decodable by Xarray, round down to hourly)
- Replace the lat, lon coordinate values with a reference grid file
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

    in_dir = f'/pscratch/sd/f/feng045/DYAMOND/{PHASE}/{runname}/{varname}/'
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/{PHASE}/{runname}/envs/'
    out_filename = f'{out_dir}{PHASE}_{runname}_{varname}.nc'
    os.makedirs(out_dir, exist_ok=True)

    # Start time for encoding
    if PHASE == 'Summer':
        start_datetime = '2016-08-01 00:00:00'
    elif PHASE == 'Winter':
        start_datetime = '2020-01-20 00:00:00' 

    # Get inputs from configuration file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)
    # Input filename
    in_filename = f"{in_dir}{config[runname][f'filename_{varname}']}"
    # Coordinate names
    xcoord_name = config[runname]['xcoord_name']
    ycoord_name = config[runname]['ycoord_name']
    varname_orig = config[runname][varname]

    # Reference grid
    ref_grid = '/pscratch/sd/f/feng045/DYAMOND/maps/IMERG_landmask_180W-180E_60S-60N.nc'

    # Read lon from reference grid
    dsref = xr.open_dataset(ref_grid)
    lon_ref = dsref['lon']
    lat_ref = dsref['lat']

    # Read input file
    print(f'Reading input file: {in_filename}')
    ds = xr.open_dataset(in_filename)

    # Rename 'xtime' coordinate (e.g., MPAS)
    if 'xtime' in ds.coords:
        # ds = ds.drop_vars(['time'])
        ds = ds.rename({'xtime':'time'})
        # Check non-standard time units
        if ds['time'].units == 'day as %Y%m%d.%f':
            # Separate the integer and decimal part of the time
            time_decimal, time_integer = np.modf(ds['time'].data)
            # Convert to Pandas DatetimeIndex
            time_decode = pd.to_datetime(time_integer, format='%Y%m%d') + pd.to_timedelta(time_decimal, unit='D')
            # Replace the time coordinate in the DataSet
            ds['time'] = time_decode

    # Change time coordinate encoding
    ds['time'].encoding['units'] = f'hours since {start_datetime}'

    # Round down the time coordinates to the nearest hour
    ds['time'] = ds['time'].dt.floor('h')
    # import pdb; pdb.set_trace()
    
    # Replace the coordinates
    ds = ds.assign_coords({xcoord_name: lon_ref, ycoord_name: lat_ref})
    ds['lon'].attrs = lon_ref.attrs
    ds['lon'].attrs['valid_min'] = np.min(lon_ref).item()
    ds['lon'].attrs['valid_max'] = np.max(lon_ref).item()
    ds['lat'].attrs = lat_ref.attrs
    ds['lat'].attrs['valid_min'] = np.min(lat_ref).item()
    ds['lat'].attrs['valid_max'] = np.max(lat_ref).item()

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