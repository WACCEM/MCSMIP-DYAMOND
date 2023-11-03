"""
Combine PyFLEXTRKR MCS pixel-level files to create a single MCS mask file for MCSMIP.
"""
import time, glob, os, sys
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster

#-------------------------------------------------------------------------
def subset_vars(infiles):
    # List of variables to exclude
    drop_varlist = [
        'base_time', 'latitude', 'longitude',
        'cloudtype', 'cloudnumber', 'nfeatures',
        'tb', 'precipitation',
        'pf_number', 'convcold_cloudnumber_orig', 'cloudnumber_orig',
        # 'tracknumber',
        'merge_tracknumber', 'split_tracknumber', 'track_status',
        'cloudmerge_tracknumber', 'cloudsplit_tracknumber', 'pcptracknumber',
    ]

    # Read Input files
    print(f'Reading input files from: {in_dir }')
    ds = xr.open_mfdataset(infiles, drop_variables=drop_varlist, mask_and_scale=False, chunks={'time': 'auto'})
    # ds = xr.open_mfdataset(infiles, drop_variables=drop_varlist, mask_and_scale=False, chunks=None)
    print(f'Done reading input files.')

    # Change time coordinate encoding
    ds['time'].encoding['units'] = f'hours since {start_datetime}'
    # Update time attributes
    ds['time'].attrs['long_name'] = 'Time'

    # Read lon from reference grid
    dsref = xr.open_dataset(ref_grid)
    lon_ref = dsref['lon']
    lat_ref = dsref['lat']
    # Make 2D lat/lon
    lon2d_ref, lat2d_ref = np.meshgrid(lon_ref, lat_ref)
    lon2d_attrs = lon_ref.attrs
    lat2d_attrs = lat_ref.attrs
    # Convert to DataArrays
    lon2d_ref = xr.DataArray(lon2d_ref, dims=('lat', 'lon'), attrs=lon2d_attrs)
    lat2d_ref = xr.DataArray(lat2d_ref, dims=('lat', 'lon'), attrs=lat2d_attrs)

    # Replace the lat/lon
    ds.update({'longitude': lon2d_ref, 'latitude': lat2d_ref})
    ds.update({'lon': lon_ref, 'lat': lat_ref})

    # Rename output variables
    rename_dict = {
        'cloudtracknumber': 'mcs_mask',
        'tracknumber': 'mcs_mask_no_mergesplit',
        'feature_number': 'cloud_mask',
    }
    dsout = ds.rename_vars(rename_dict)
    # Update variable attributes
    dsout['mcs_mask'].attrs = {
        'long_name': 'MCS mask with track number (required)',
        'units': 'unitless',
    }
    dsout['mcs_mask_no_mergesplit'].attrs = {
        'long_name': 'MCS mask with track number excluding merging/splitting (optional)',
        'units': 'unitless',
    }
    dsout['cloud_mask'].attrs = {
        'long_name': 'All cloud object mask (optional)',
        'units': 'unitless',
    }

    # Replace global attributes
    gattr_dict = {
        'Title': f'{PHASE} {runname} MCS mask file',
        'tracker': 'PyFLEXTRKR',
        'Contact':'Zhe Feng, zhe.feng@pnnl.gov', 
        'Created_on':time.ctime(time.time()),
    }
    dsout.attrs = gattr_dict

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netCDF
    print(f'Writing output ...')
    dsout.to_netcdf(path=out_filename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'Output saved: {out_filename}')

    return


if __name__=='__main__':

    PHASE = sys.argv[1]
    runname = sys.argv[2]

    if PHASE == 'Summer':
        start_datetime = '2016-08-01 00:00:00'
        period = '20160801.0000_20160910.0000'
    elif PHASE == 'Winter':
        start_datetime = '2020-01-20 00:00:00'
        period = '20200120.0000_20200301.0000'
    
    in_dir = f'/pscratch/sd/f/feng045/DYAMOND/{PHASE}/{runname}/mcstracking/{period}/'
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask/{PHASE}/PyFLEXTRKR/'
    ref_grid = '/pscratch/sd/f/feng045/DYAMOND/maps/IMERG_landmask_180W-180E_60S-60N.nc'

    # Input/output file basename
    in_basename = 'mcstrack_'
    out_filename = f'{out_dir}mcs_mask_{PHASE}_{runname}.nc'

    print(f'Start time: {time.ctime(time.time())}')
    os.makedirs(out_dir, exist_ok=True)

    # Set flag to run in parallel (1) or serial (0)
    run_parallel = 1
    # Number of workers for Dask
    n_workers = 128
    # Threads per worker
    threads_per_worker = 1

    # Find all input files
    input_files = sorted(glob.glob(f'{in_dir}{in_basename}*.nc'))
    # import pdb; pdb.set_trace()

    if run_parallel==1:
        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)
        results = []

    # Run combine
    outfile = subset_vars(input_files)
    print(f'End time: {time.ctime(time.time())}')

    if run_parallel==1:
        # Close the Dask cluster
        client.close()
        cluster.close()
