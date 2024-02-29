"""
Prepare ERA5 grid file
- Rename lat, lon coordinate and inverse lat order
- Roll longitude to make the grid start from -180~+180
"""
import numpy as np
import glob, sys, os
import xarray as xr
import yaml

if __name__ == "__main__":

    in_dir = '/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.invariant/197901/'
    in_filename = f'{in_dir}e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc'

    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/maps/'
    out_filename = f'{out_dir}era5_landmask.nc'

    xcoord_name = 'longitude'
    ycoord_name = 'latitude'
    # lat_bounds = [-60., 60.]
    lat_bounds = [-59.75, 59.75]

    # Make longitude grid from -180~+179.75
    lon180 = np.arange(-180, 180, 0.25)

    # Read input file
    ds = xr.open_dataset(in_filename)
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
    dsout = ds.rename({'LSM': 'landseamask'})

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    print(f'Writing output ...')
    dsout.to_netcdf(path=out_filename, mode="w", format="NETCDF4", unlimited_dims='time', encoding=encoding)
    print(f"{out_filename}")