import numpy as np
import xarray as xr
import pandas as pd
import sys, os
import time
import yaml

if __name__ == "__main__":

    PHASE = sys.argv[1]
    runname = sys.argv[2]
    varname = sys.argv[3]
    start_date = sys.argv[4]
    end_date = sys.argv[5]

    # Config file contains input filenames for each model & environmental variable names
    # config_file = f'/global/homes/f/feng045/program/mcsmip/dyamond/src/config_env_files_{PHASE}.yml'

    # Inputs
    env_dir = f'/pscratch/sd/f/feng045/DYAMOND/{PHASE}/{runname}/envs/'
    env_file = f"{env_dir}{PHASE}_{runname}_{varname}.nc"
    # Outputs 
    out_dir = env_dir
    out_filename_timeseries = f'{out_dir}{PHASE}_{runname}_{varname}_timeseries.nc'
    out_filename_map = f'{out_dir}{PHASE}_{runname}_{varname}_map.nc'

    # Specify regions and time periods for averaging
    if PHASE == 'Summer':
        datetime_range = pd.to_datetime([start_date, end_date])
        lon_bounds = [-180, 180]
        lat_bounds = [-15, 30]
    if PHASE == 'Winter':
        datetime_range = pd.to_datetime([start_date, end_date])
        lon_bounds = [-180, 180]
        lat_bounds = [-20, 15]

    # Get inputs from configuration file
    # stream = open(config_file, 'r')
    # config = yaml.full_load(stream)

    # Read input data
    ds = xr.open_dataset(env_file)
    ntimes = ds.dims['time']
    longitude = ds['lon']
    latitude = ds['lat']
    # Get total number of grids
    ny = ds.sizes['lat']
    nx = ds.sizes['lon']

    # Average within specified region to get time series
    print(f'Calculating spatial average ...')
    VAR_avg_timeseries = ds[varname].sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    ).mean(dim=('lon', 'lat'), keep_attrs=True)

    # Average between specified time period
    print(f'Calculating temporal average ...')
    VAR_avg_map = ds[varname].sel(time=slice(datetime_range[0], datetime_range[1])).mean(dim='time', keep_attrs=True)

    #-------------------------------------------------------------------------
    # Write output time series file
    #-------------------------------------------------------------------------
    var_dict = {
        varname: (['time'], VAR_avg_timeseries.data),
    }
    coord_dict = {
        'time': (['time'], ds['time'].data),
    }
    gattr_dict = {
        'title': f'Mean {varname} time series',
        'contact':'Zhe Feng, zhe.feng@pnnl.gov',
        'lon_bounds': lon_bounds,
        'lat_bounds': lat_bounds,
        'created_on':time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    fillvalue = np.nan
    # Set encoding/compression for all variables
    comp = dict(zlib=True, _FillValue=fillvalue)
    encoding = {var: comp for var in dsout.data_vars}
    dsout.to_netcdf(path=out_filename_timeseries, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'Time series saved: {out_filename_timeseries}')

    #-------------------------------------------------------------------------
    # Write output map file
    #-------------------------------------------------------------------------
    var_dict = {
        varname: (['lat', 'lon'], VAR_avg_map.data),
    }
    coord_dict = {
        # 'time': (['time'], ds['time'].data),
        'lat': (['lat'], latitude.data, latitude.attrs),
        'lon': (['lon'], longitude.data, longitude.attrs),
    }
    gattr_dict = {
        'title': f'Time mean {varname} map',
        'contact':'Zhe Feng, zhe.feng@pnnl.gov',
        'start_date': datetime_range[0].strftime('%Y-%m-%dT%H'),
        'end_date': datetime_range[1].strftime('%Y-%m-%dT%H'),
        'created_on':time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
    fillvalue = np.nan
    # Set encoding/compression for all variables
    comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    dsout.to_netcdf(path=out_filename_map, mode='w', format='NETCDF4', encoding=encoding)
    print(f'Map saved: {out_filename_map}')

    # import pdb; pdb.set_trace()