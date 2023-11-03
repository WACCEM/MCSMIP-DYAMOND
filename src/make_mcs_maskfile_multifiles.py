"""
Create example MCS mask files for MCSMIP from PyFLEXTRKR pixel files.
Individual hourly mask files can be combined to a single file using NCO:
>ncrcat -h mcsmask*nc mcs_mask.nc
"""
import time, glob, os
import xarray as xr
import pandas as pd
import dask
from dask.distributed import Client, LocalCluster

def subset_vars(infile):

    drop_varlist = [
        'base_time', 'latitude', 'longitude',
        'cloudtype', 'cloudnumber', 'nfeatures',
        'tb', 'precipitation',
        'pf_number', 'convcold_cloudnumber_orig', 'cloudnumber_orig',
        'tracknumber',
        'merge_tracknumber', 'split_tracknumber', 'track_status',
        'cloudmerge_tracknumber', 'cloudsplit_tracknumber', 'pcptracknumber',
    ]

    # Read Input data 
    ds = xr.open_dataset(infile, drop_variables=drop_varlist, mask_and_scale=False)

    # Convert 'time' to pandas datetime format
    time_in_datetime = pd.to_datetime(ds['time'].values)
    # Make output file time string
    outfile_timestring = time_in_datetime.strftime("%Y%m%d_%H%M%S").item()
    # Make output filename
    outfile = f'{out_dir}{out_basename}{outfile_timestring}.nc'

    # Change time coordinate encoding
    ds['time'].encoding['units'] = f'hours since {start_datetime}'
    # Update time attributes
    ds['time'].attrs['long_name'] = 'Time'
    # ds['time'].attrs['units'] = f'hours since {start_datetime}'

    # Rename output variables
    rename_dict = {
        'cloudtracknumber':'mcs_mask',
        'feature_number':'cloud_mask',
    }
    dsout = ds.rename_vars(rename_dict)
    # Update variable attributes
    dsout['mcs_mask'].attrs = {
        'long_name': 'MCS mask with track number (required)',
        'units': 'unitless',
    }
    dsout['cloud_mask'].attrs = {
        'long_name': 'Cloud object mask (optional)',
        'units': 'unitless',
    }

    # Replace global attributes
    gattr_dict = {
        'Title': 'Example MCS mask file',
        'Contact':'Zhe Feng, zhe.feng@pnnl.gov', 
        'Created_on':time.ctime(time.time()),
    }
    dsout.attrs = gattr_dict

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}
    # Update double type coordinate variables to float
    for var in dsout.coords:
        if (dsout[var].dtype=='float64') & (var != 'time'):
            encoding.update({var:{'zlib':True,'dtype':'float32'}})

    # Write to netCDF
    dsout.to_netcdf(path=outfile, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'Output saved: {outfile}')

    return outfile


if __name__=='__main__':

    runname = 'NICAM'
    in_dir = f'/pscratch/sd/f/feng045/DYAMOND/Summer/{runname}/mcstracking/20160801.0000_20160910.0000/'
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask/Summer/{runname}/hourly/'

    os.makedirs(out_dir, exist_ok=True)

    start_datetime = '2016-08-01 00:00:00'

    # Set flag to run in parallel (1) or serial (0)
    run_parallel = 1
    # Number of workers for Dask
    n_workers = 128
    # Threads per worker
    threads_per_worker = 1

    # Input/output file basename
    in_basename = 'mcstrack_'
    out_basename = 'mcsmask_'
    # Find all input files
    input_files = sorted(glob.glob(f'{in_dir}{in_basename}*.nc'))


    if run_parallel==1:
        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)
        results = []

    # Loop over each intput file
    for ifile in input_files:
        if run_parallel==0:
            # serial version
            outfile = subset_vars(ifile)
        elif run_parallel==1:
            # parallel version
            status = dask.delayed(subset_vars)(ifile)
            results.append(status)

    if run_parallel==1:
        # Compute results from Dask
        results = dask.compute(*results)
