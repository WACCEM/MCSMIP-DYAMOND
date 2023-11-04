"""
Calculate monthly total, MCS precipitation amount and frequency, save output to a netCDF file.
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
    pixel_dir = config['pixeltracking_outpath']
    pcp_thresh = config['pcp_thresh']
    mask_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask/{PHASE}/{tracker}/'
    output_monthly_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_stats/{PHASE}/{tracker}/'
    pixel_basename = 'mcstrack_'
    pcpvarname = 'precipitation'

    # Mask filename
    mask_file = f'{mask_dir}mcs_mask_{PHASE}_{runname}.nc'

    # Check required input files
    if os.path.isfile(mask_file) == False:
        print(f'ERROR: mask file does not exist: {mask_file}')
        sys.exit(f'Code will exist now.')
    # if os.path.isfile(config_file) == False:
    #     print(f'ERROR: config file does not exist: {config_file}')
    #     sys.exit(f'Code will exist now.')

    # Generate time marks within the start/end datetime
    all_datetimes = pd.date_range(start=start_datetime, end=end_datetime, freq='1H')
    file_datetimes = all_datetimes.strftime('%Y%m%d_%H')

    # Output file name
    sdate = all_datetimes.min().strftime('%Y%m%d')
    edate = all_datetimes.max().strftime('%Y%m%d')
    output_filename = f'{output_monthly_dir}mcs_rainmap_{PHASE}_{runname}.nc'

    # Find all precipitation files within the start/end datetime
    pcpfiles = []
    for tt in range(0, len(file_datetimes)):
        pcpfiles.extend(sorted(glob.glob(f'{pixel_dir}{pixel_basename}{file_datetimes[tt]}*.nc')))
    nfiles = len(pcpfiles)
    print(pixel_dir)
    print('Number of files: ', nfiles)
    os.makedirs(output_monthly_dir, exist_ok=True)


    if nfiles > 0:

        # Read precipitation files and concatinate
        ds = xr.open_mfdataset(pcpfiles, concat_dim='time', combine='nested')
        print('Finish reading input files.')
        ntimes_pcp = ds.dims['time']
        longitude = ds['longitude'].isel(time=0)
        latitude = ds['latitude'].isel(time=0)

        # Read MCS mask file
        print(f'Reading MCS mask file: {mask_file}')
        dsm = xr.open_dataset(mask_file, mask_and_scale=False)

        # # Check duplicate times in the mask DataSet
        # duplicates = dsm.indexes['time'].duplicated()
        # if duplicates.any() == True:
        #     # Group by time and take the first value for each group (remove duplicates)
        #     dsm_unique_times = dsm.groupby('time').first()
        #     # Resetting the index to get a new time coordinate
        #     dsm = dsm_unique_times.reset_index('time').set_xindex('time')
        # import pdb; pdb.set_trace()

        # Subset mask file for the times
        dsm = dsm.sel(time=ds['time'], method='nearest')
        # dsm = dsm.reset_index('time', drop=True)
        # dsm = dsm.sel(time=slice(ds['time'].min(), ds['time'].max()))
        ntimes_mask = dsm.sizes['time']
        if (ntimes_mask != ntimes_pcp):
            print(f'ERROR: Subset times in mcs_mask file is NOT the same with precipitation file.')
            sys.exit(f'Code will exit now.')
        # Replace mask DataSet coordinates with that from the precipitation DataSet
        dsm = dsm.assign_coords({'time': ds['time']})
        # Replace precipitation DataSet lat/lon coordinates
        ds = ds.assign_coords({'lon': dsm['lon'], 'lat': dsm['lat']})
        # dsm = dsm.assign_coords({'lon': ds['lon'], 'lat': ds['lat'], 'time': ds['time']})
        # time_mcs_mask = dsm['time']
        mcs_mask = dsm['mcs_mask']
        
        # Sum MCS precipitation over time, use cloudtracknumber > 0 as mask
        mcsprecip = ds[pcpvarname].where(mcs_mask > 0).sum(dim='time')

        # Sum total precipitation over time
        totprecip = ds[pcpvarname].sum(dim='time')

        # Sum MCS counts over time to get number of hours
        mcscloudct = (mcs_mask > 0).sum(dim='time')
        # Count hours MCS precipitation > pcp_thresh
        mcspcpct = (ds[pcpvarname].where(mcs_mask > 0) > pcp_thresh).sum(dim='time')


        ############################################################################
        # Write output file
        var_dict = {
            'longitude': (['lat', 'lon'], longitude.data, longitude.attrs),
            'latitude': (['lat', 'lon'], latitude.data, latitude.attrs),
            'precipitation': (['time', 'lat', 'lon'], totprecip.expand_dims('time', axis=0).data),
            'mcs_precipitation': (['time', 'lat', 'lon'], mcsprecip.expand_dims('time', axis=0).data),
            'mcs_precipitation_count': (['time', 'lat', 'lon'], mcspcpct.expand_dims('time', axis=0).data),
            'mcs_cloud_count': (['time', 'lat', 'lon'], mcscloudct.expand_dims('time', axis=0).data),
            'ntimes': (['time'], xr.DataArray(ntimes_mask).expand_dims('time', axis=0).data),
        }
        coord_dict = {
            'time': (['time'], [all_datetimes[0]]),
            'lat': (['lat'], ds['lat'].data),
            'lon': (['lon'], ds['lon'].data),
        }
        gattr_dict = {
            'title': f'MCS precipitation accumulation from {tracker}',
            'contact':'Zhe Feng, zhe.feng@pnnl.gov',
            'start_date': sdate,
            'end_date': edate,
            'created_on':time.ctime(time.time()),
        }
        dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
        # dsout.time.attrs['long_name'] = 'Epoch Time (since 1970-01-01T00:00:00)'
        # dsout.time.attrs['units'] = 'Seconds since 1970-1-1 0:00:00 0:00'
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

        dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
        print(f'Output saved: {output_filename}')
        print(f'End time: {time.ctime(time.time())}')

    else:
        print(f'No files found. Code exits.')
