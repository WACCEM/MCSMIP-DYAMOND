"""
Subset and roll the input tracking data either from -180~180 to 0~360, or from 0~360 to -180~+180.

Author: Zhe Feng, zhe.feng@pnnl.gov
History:
07/12/2023 - Written.
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

    # Check lon limits
    lon_min = ds['lon'].min().data
    if (lon_min >= 0):
        # Read lon from reference grid
        dsref = xr.open_dataset(ref_grid)
        lon180 = dsref['lon']
        # import pdb; pdb.set_trace()

        # Roll lon=1800 to make the data start from -180~180
        ds = ds.roll(lon=1800, roll_coords=True)
        # Convert longitude coordinates from 0~360 to -180~+180
        # lon180 = ((ds['lon'].data - 180) % 360) - 180
        ds = ds.assign_coords(lon=lon180)
        ds['lon'].attrs = lon_attrs
        ds['lon'].attrs['valid_min'] = -180.
        ds['lon'].attrs['valid_max'] = 180.

    elif (lon_min < 0):
        # Roll lon=1800 to make the data start from 0~360
        ds = ds.roll(lon=1800, roll_coords=True)
        # Convert longitude coordinates from -180~180 to 0~360
        lon360 = ds['lon'].data % 360
        ds = ds.assign_coords(lon=lon360)
        ds['lon'].attrs = lon_attrs
        ds['lon'].attrs['valid_min'] = 0.
        ds['lon'].attrs['valid_max'] = 360.

    else:
        print(f'ERROR: unexpected lon min: {lon_min}')
        sys.exit()

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
    in_dir = f'/pscratch/sd/f/feng045/DYAMOND/{phase}/{run_name}/olr_pcp_instantaneous/'
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/{phase}/{run_name}/olr_pcp/'

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