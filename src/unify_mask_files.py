"""
Unify the MCS mask files from different trackers:
- Rename the coordinate names to: lon, lat
- Replace the lat, lon coordinate values with a reference grid file
"""
import numpy as np
import glob, sys, os
import time
import xarray as xr
import pandas as pd

if __name__ == "__main__":
    
    PHASE = sys.argv[1]
    runname = sys.argv[2]
    tracker = sys.argv[3]

    in_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask/{PHASE}/{tracker}/orig/'
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask/{PHASE}/{tracker}/'
    out_filename = f'{out_dir}mcs_mask_{PHASE}_{runname}.nc'
    
    if PHASE == 'Summer':
        start_datetime = '2016-08-01 00:00:00'
        # period = '20160801.0000_20160910.0000'
    elif PHASE == 'Winter':
        start_datetime = '2020-01-20 00:00:00'
        # period = '20200120.0000_20200301.0000'

    # Mask filename
    if tracker == 'PyFLEXTRKR':
        mask_file = f'{in_dir}mcs_mask_{runname}.nc'
        xcoord_name = 'lon'
        ycoord_name = 'lat'
    elif tracker == 'MOAAP':
        mask_file = f'{in_dir}mcs_mask_{runname}_{PHASE}.nc'
        xcoord_name = 'x'
        ycoord_name = 'y'
    elif tracker == 'TOOCAN':
        mask_file = f'{in_dir}mcs_mask_{PHASE}_{runname}.nc'
        xcoord_name = 'longitude'
        ycoord_name = 'latitude'
    elif tracker == 'tobac':
        phase = PHASE.lower()
        mask_file = f'{in_dir}tobac_{runname}_{phase}_MCS_mask_file.nc'
        xcoord_name = 'lon'
        ycoord_name = 'lat'
    elif tracker == 'TAMS':
        mask_file = f'{in_dir}mcs_mask_{PHASE}_{runname}.nc'
        xcoord_name = 'lon'
        ycoord_name = 'lat'
    elif tracker == 'simpleTrack':
        mask_file = f'{in_dir}DYAMOND_{PHASE}_{runname}_MCS_masks.nc'
        xcoord_name = 'lon'
        ycoord_name = 'lat'
    else:
        print(f'ERROR: {tracker} file format is undefined.')
        print(f'Code will exist now.')
        sys.exit()
    
    # Reference grid
    ref_grid = '/pscratch/sd/f/feng045/DYAMOND/maps/IMERG_landmask_180W-180E_60S-60N.nc'

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

    # Read MCS mask file
    print(f'Reading MCS mask file: {mask_file}')
    ds = xr.open_dataset(mask_file, mask_and_scale=False)

    # # The time coordinate in the simpleTrack Winter OBS file is wrong, replace it
    # if (PHASE == 'Winter') & (runname == 'OBS') & (tracker == 'simpleTrack'):
    #     hours = pd.date_range(start=start_datetime, periods=len(ds.time), freq='H')
    #     ds = ds.assign_coords(time=hours)

    # Rename 'xtime' coordinate (e.g., MPAS)
    if 'xtime' in ds.coords:
        ds = ds.drop_vars(['time'])
        ds = ds.rename({'xtime':'time'})
        # import pdb; pdb.set_trace()

    # Check duplicate times in the mask DataSet
    duplicates = ds.indexes['time'].duplicated()
    if duplicates.any() == True:
        # Group by time and take the first value for each group (remove duplicates)
        ds_unique_times = ds.groupby('time').first()
        # Resetting the index to get a new time coordinate
        ds = ds_unique_times.reset_index('time').set_xindex('time')

    # Change time coordinate encoding
    ds['time'].encoding['units'] = f'hours since {start_datetime}'

    # Rename the original coordinates
    ds = ds.rename({xcoord_name: 'lon', ycoord_name: 'lat'})
    # import pdb; pdb.set_trace()

    # Replace the coordinates
    ds = ds.assign_coords({xcoord_name: lon_ref, ycoord_name: lat_ref})
    ds['lon'].attrs = lon_ref.attrs
    ds['lon'].attrs['valid_min'] = np.min(lon_ref).item()
    ds['lon'].attrs['valid_max'] = np.max(lon_ref).item()
    ds['lat'].attrs = lat_ref.attrs
    ds['lat'].attrs['valid_min'] = np.min(lat_ref).item()
    ds['lat'].attrs['valid_max'] = np.max(lat_ref).item()

    # Replace the 2D lat/lon
    ds.update({'longitude': lon2d_ref})
    ds.update({'latitude': lat2d_ref})

    # Drop original coordinates
    # if (tracker != 'TOOCAN'):
    #     dsout = ds.drop_vars([xcoord_name, ycoord_name])
    # else:
    #     dsout = ds
    if (tracker == 'TOOCAN') | (tracker == 'tobac') | (tracker == 'TAMS') | (tracker == 'simpleTrack'):
        dsout = ds
    else:
        dsout = ds.drop_vars([xcoord_name, ycoord_name])
    # import pdb; pdb.set_trace()

    # Drop variables
    if tracker == 'tobac':
        coordinates_to_drop = ['feature', 'cell', 'track']
        variables_to_drop = ['all_feature_labels', 'mcs_feature_labels', 'all_cell_labels', 'mcs_cell_labels']
    elif tracker == 'TAMS':
        coordinates_to_drop = ['mcs_id']
        variables_to_drop = []
    else:
        coordinates_to_drop = []
        variables_to_drop = []
    if len(coordinates_to_drop) > 0:
        # Drop all variables associated with the specified coordinates
        for coord in coordinates_to_drop:
            dsout = dsout.drop_vars([var for var in dsout.variables if coord in dsout[var].dims])
    if len(variables_to_drop) > 0:
        dsout = dsout.drop_vars(variables_to_drop)

    # Rename variables
    if tracker == 'tobac':
        dsout = dsout.rename({'mcs_track_labels':'mcs_mask'})

    # Convert mcs_mask data type to int
    if dsout['mcs_mask'].dtype != int:
        dsout['mcs_mask'] = dsout['mcs_mask'].astype(int)
        # Replace negative (missing) values with 0
        dsout['mcs_mask'] = dsout['mcs_mask'].where(dsout['mcs_mask'] > 0, other=0)
        dsout['mcs_mask'].attrs['_FillValue'] = 0

    # Round down the time coordinates to the nearest hour
    dsout['time'] = dsout['time'].dt.floor('H')
    # import pdb; pdb.set_trace()

    # Replace global attributes
    gattrs = {
        'Title': f'{PHASE} {runname} MCS mask file',
        'tracker': f'{tracker}',
        'Created_on': time.ctime(time.time()),
    }
    dsout.attrs = gattrs
    # Add global attributes
    # dsout.attrs['Title'] = f'{PHASE} {runname} MCS mask file'
    # dsout.attrs['tracker'] = f'{tracker}'
    # dsout.attrs['Created_on'] = time.ctime(time.time())
    # Delete attributes
    # dsout.attrs.pop('history')
    # import pdb; pdb.set_trace()

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    print(f'Writing output ...')
    dsout.to_netcdf(path=out_filename, mode="w", format="NETCDF4", unlimited_dims='time', encoding=encoding)
    print(f"{out_filename}")