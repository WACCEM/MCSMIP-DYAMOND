"""
Replaces OLR data lat/lon with that from a reference grid file.
This fixes the issue of the input lat/lon that is slightly different among some DYAMOND files (e.g., Winter ICON),
causing Xarray issue when reading the files using open_mfdataset().

Author: Zhe Feng, zhe.feng@pnnl.gov
History:
08/29/2023 - Written.
"""
import numpy as np
import glob, os, sys
import xarray as xr
import pandas as pd
import dask
from dask.distributed import Client, LocalCluster

#-----------------------------------------------------------------------
def subset_data(infiles, out_dir, minlat, maxlat):
    # print('Reading input data ... ')
    ds = xr.open_dataset(infiles, decode_times=False)
    # Subset 60S-60N
    ds = ds.sel(lat=slice(minlat, maxlat))
    lon_attrs = ds['lon'].attrs
    lat_attrs = ds['lat'].attrs
    # lon2d_attrs = ds['longitude'].attrs
    # lat2d_attrs = ds['latitude'].attrs

    # # Check lon limits
    # lon_min = ds['lon'].min().data

    # Read lon from reference grid
    dsref = xr.open_dataset(ref_grid)
    dsref = dsref.sel(lat=slice(minlat, maxlat))
    lon_ref = dsref['lon']
    lat_ref = dsref['lat']
    # # Make 2D lat/lon
    # lon2d_ref, lat2d_ref = np.meshgrid(lon_ref, lat_ref)
    # # Convert to DataArrays
    # lon2d_ref = xr.DataArray(lon2d_ref, dims=('lat', 'lon'), attrs=lon2d_attrs)
    # lat2d_ref = xr.DataArray(lat2d_ref, dims=('lat', 'lon'), attrs=lat2d_attrs)

    # Replace the coordinates
    ds = ds.assign_coords(lon=lon_ref)
    ds['lon'].attrs = lon_attrs
    ds['lon'].attrs['valid_min'] = np.min(lon_ref).item()
    ds['lon'].attrs['valid_max'] = np.max(lon_ref).item()

    ds = ds.assign_coords(lat=lat_ref)
    ds['lat'].attrs = lat_attrs
    ds['lat'].attrs['valid_min'] = minlat
    ds['lat'].attrs['valid_max'] = maxlat

    # Replace the 2D lat/lon
    # ds.update({'longitude': lon2d_ref})
    # ds.update({'latitude': lat2d_ref})
    # import pdb; pdb.set_trace()

    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in ds.data_vars}
    # Update time variable dtype as 'double' for better precision
    bt_dict = {'time': {'zlib':True, 'dtype':'float64'}}
    encoding.update(bt_dict)

    # Write to netcdf file
    filename_out = out_dir + os.path.basename(infiles)
    ds.to_netcdf(path=filename_out, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'Saved as: {filename_out}')

    # import pdb; pdb.set_trace()
    status = 1
    return status


if __name__=='__main__':

    # Get inputs
    phase = sys.argv[1]
    run_name = sys.argv[2]

    # Set flag to run in parallel (1) or serial (0)
    run_parallel = 1
    # Number of workers for Dask
    n_workers = 128
    # Threads per worker
    threads_per_worker = 1

    if phase == 'Summer':
        # Phase-I (Summer)
        start_date = '2016-08-01'
        end_date = '2020-09-11'
    elif phase == 'Winter':
        # Phase-II (Winter)
        start_date = '2020-01-20'
        end_date = '2020-03-01'

    # Latitude subset range
    minlat, maxlat = -60., 60.

    # Input/output directory
    # in_dir = f'/pscratch/sd/f/feng045/DYAMOND/{phase}/{run_name}/olr_pcp_instantaneous/'
    # out_dir = f'/pscratch/sd/f/feng045/DYAMOND/{phase}/{run_name}/olr_pcp/'

    in_dir = f'/pscratch/sd/f/feng045/DYAMOND/{phase}/{run_name}/olr_pcp_instantaneous_orig/'
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/{phase}/{run_name}/olr_pcp_instantaneous/'

    # Reference grid
    ref_grid = '/pscratch/sd/f/feng045/DYAMOND/Summer/OBS/olr_pcp/merg_2016080100_4km-pixel.nc'

    # Phase-II (Winter)
    # in_dir = f'/pscratch/sd/f/feng045/Winter/{run_name}/olr_pcp/'
    # out_dir = f'/pscratch/sd/f/feng045/Winter/{run_name}/olr_pcp_180/'

    os.makedirs(out_dir, exist_ok=True)

    # Create a range of dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates.strftime('%Y%m%d')
    # Find all files from the dates    
    infiles = []
    for ii in range(0, len(dates)):
        infiles.extend(sorted(glob.glob(f'{in_dir}*{dates[ii]}*.nc')))
    nf1 = len(infiles)

    if run_parallel==0:
        # serial version
        for ifile in range(nf1):
            print(infiles[ifile])
            status = subset_data(infiles[ifile], out_dir, minlat, maxlat)

    elif run_parallel==1:
        # parallel version
            
        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)

        results = []
        for ifile in range(nf1):
            print(infiles[ifile])

            # Call subset function
            status = dask.delayed(subset_data)(infiles[ifile], out_dir, minlat, maxlat)
            results.append(status)

        # Collect results from Dask
        # print("Precompute step")
        results = dask.compute(*results)