"""
Calculate total, MCS precipitation amount and frequency within a period, save output to a netCDF file.
"""
import numpy as np
import glob, sys, os
import xarray as xr
import pandas as pd
import time, datetime, calendar, pytz
from pyflextrkr.ft_utilities import load_config

if __name__ == "__main__":

    # Get inputs from command line
    config_file = sys.argv[1]
    tracker = sys.argv[2]
    start_datetime = sys.argv[3]
    end_datetime = sys.argv[4]

    print(f'Start time: {time.ctime(time.time())}')

    # Get runname and PHASE from config_file name
    parts = config_file.split("/")
    config_file_basename = parts[-1]
    # Config filname format: config_dyamond_PHASE_runname.yml
    runname = config_file_basename.split("_")[-1].split(".")[0]
    PHASE = config_file_basename.split("_")[-2].capitalize()
    print(f'{PHASE} {runname} {tracker}')

    # Get inputs from configuration file
    config = load_config(config_file)
    # pixel_dir = config['pixeltracking_outpath']
    pcp_thresh = config['pcp_thresh']
    mask_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask/{PHASE}/{tracker}/'
    pcp_dir = f'/pscratch/sd/f/feng045/DYAMOND/OLR_Precipitation_combined/'
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_stats/{PHASE}/{tracker}/'
    pixel_basename = 'mcstrack_'
    pcpvarname = 'precipitation'

    # Input filenames
    mask_file = f'{mask_dir}mcs_mask_{PHASE}_{runname}.nc'
    pcp_file = f'{pcp_dir}olr_pcp_{PHASE}_{runname}.nc'

    # Output file name
    output_filename = f'{out_dir}mcs_rainmap_{PHASE}_{runname}.nc'
    os.makedirs(out_dir, exist_ok=True)

    # Check required input files
    if os.path.isfile(mask_file) == False:
        print(f'ERROR: mask file does not exist: {mask_file}')
        sys.exit(f'Code will exist now.')
    if os.path.isfile(pcp_file) == False:
        print(f'ERROR: pcp file does not exist: {pcp_file}')
        sys.exit(f'Code will exist now.')

    # Read precipitation files
    ds = xr.open_dataset(pcp_file)
    # Subset times
    ds = ds.sel(time=slice(start_datetime, end_datetime))

    # Convert CFTimeIndex to Pandas DatetimeInex
    # This gets around issues with time coordinates in cftime.DatetimeNoLeap format (e.g., SCREAM)
    if runname == 'SCREAM':
        ds_datetimeindex = ds.indexes['time'].to_datetimeindex()
        # Replace the original time coordinate
        ds = ds.assign_coords({'time': ds_datetimeindex})
    ntimes_pcp = ds.dims['time']
    out_time = ds['time'].isel(time=0)
    lon = ds['lon']
    lat = ds['lat']

    # Read MCS mask file
    print(f'Reading MCS mask file: {mask_file}')
    dsm = xr.open_dataset(mask_file, mask_and_scale=False)
    # Convert CFTimeIndex to Pandas DatetimeInex
    if dsm['time'].encoding.get('calendar') == 'noleap':
        dsm_datetimeindex = dsm.indexes['time'].to_datetimeindex()
        # Replace the original time coordinate
        dsm = dsm.assign_coords({'time': dsm_datetimeindex})

    # Subset mask file for the times
    dsm = dsm.sel(time=ds['time'], method='nearest')
    ntimes_mask = dsm.sizes['time']
    if (ntimes_mask != ntimes_pcp):
        print(f'ERROR: Subset times in mcs_mask file is NOT the same with precipitation file.')
        sys.exit(f'Code will exit now.')

    # Replace mask DataSet coordinates with that from the precipitation DataSet
    dsm = dsm.assign_coords({'time': ds['time']})
    # Replace precipitation DataSet lat/lon coordinates
    ds = ds.assign_coords({'lon': dsm['lon'], 'lat': dsm['lat']})
    mcs_mask = dsm['mcs_mask']
    
    # Sum MCS precipitation over time, use cloudtracknumber > 0 as mask
    mcsprecip = ds[pcpvarname].where(mcs_mask > 0).sum(dim='time')

    # Sum total precipitation over time
    totprecip = ds[pcpvarname].sum(dim='time')

    # Sum MCS counts over time to get number of hours
    mcscloudct = (mcs_mask > 0).sum(dim='time')

    # Count hours MCS precipitation > pcp_thresh
    mcspcpct = (ds[pcpvarname].where(mcs_mask > 0) > pcp_thresh).sum(dim='time')
    print(f'Done with calculations.')


    ############################################################################
    # Write output file
    var_dict = {
        'precipitation': (['time', 'lat', 'lon'], totprecip.expand_dims('time', axis=0).data),
        'mcs_precipitation': (['time', 'lat', 'lon'], mcsprecip.expand_dims('time', axis=0).data),
        'mcs_precipitation_count': (['time', 'lat', 'lon'], mcspcpct.expand_dims('time', axis=0).data),
        'mcs_cloud_count': (['time', 'lat', 'lon'], mcscloudct.expand_dims('time', axis=0).data),
        'ntimes': (['time'], xr.DataArray(ntimes_mask).expand_dims('time', axis=0).data),
    }
    coord_dict = {
        'time': (['time'], [out_time.data]),
        'lat': (['lat'], ds['lat'].data),
        'lon': (['lon'], ds['lon'].data),
    }
    gattr_dict = {
        'Title': f'MCS precipitation accumulation',
        'phase': f'{PHASE}',
        'tracker': f'{tracker}',
        'contact':'Zhe Feng, zhe.feng@pnnl.gov',
        'start_date': start_datetime,
        'end_date': end_datetime,
        'created_on':time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
    dsout['lon'].attrs['long_name'] = 'Longitude'
    dsout['lon'].attrs['units'] = 'degree'
    dsout['lat'].attrs['long_name'] = 'Latitude'
    dsout['lat'].attrs['units'] = 'degree'
    dsout['ntimes'].attrs['long_name'] = 'Number of hours during the period'
    dsout['ntimes'].attrs['units'] = 'count'
    dsout['precipitation'].attrs['long_name'] = 'Total precipitation'
    dsout['precipitation'].attrs['units'] = 'mm'
    dsout['mcs_precipitation'].attrs['long_name'] = 'MCS precipitation'
    dsout['mcs_precipitation'].attrs['units'] = 'mm'
    dsout['mcs_precipitation_count'].attrs['long_name'] = 'Number of hours MCS precipitation is recorded'
    dsout['mcs_precipitation_count'].attrs['units'] = 'hour'
    dsout['mcs_cloud_count'].attrs['long_name'] = 'Number of hours MCS cloud is recorded'
    dsout['mcs_cloud_count'].attrs['units'] = 'hour'

    fillvalue = np.nan
    # Set encoding/compression for all variables
    comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    print(f'Writing output ...')
    dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'Output saved: {output_filename}')
    print(f'End time: {time.ctime(time.time())}')

