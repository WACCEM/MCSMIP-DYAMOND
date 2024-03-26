"""
Calculate global mean total, MCS precipitation time series and save output to a netCDF file.
"""
import numpy as np
import sys, os
import xarray as xr
import pandas as pd
import time

if __name__ == "__main__":

    PHASE = sys.argv[1]
    runname = sys.argv[2]
    # start_date = sys.argv[3]
    # end_date = sys.argv[4]

    # Inputs
    tracker = 'PyFLEXTRKR'
    pcp_dir = f'/pscratch/sd/f/feng045/DYAMOND/OLR_Precipitation_combined/'
    mask_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask/{PHASE}/{tracker}/'
    rain_file = f'{pcp_dir}olr_pcp_{PHASE}_{runname}.nc'
    mask_file = f'{mask_dir}mcs_mask_{PHASE}_{runname}.nc'
    # Outputs 
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/{PHASE}/{runname}/envs/'
    out_filename_timeseries = f'{out_dir}{PHASE}_{runname}_rain_timeseries.nc'
    os.makedirs(out_dir, exist_ok=True)

    # Specify regions and time periods for averaging
    if PHASE == 'Summer':
        # datetime_range = pd.to_datetime([start_date, end_date])
        lon_bounds = [-180, 180]
        lat_bounds = [-15, 30]
    if PHASE == 'Winter':
        # datetime_range = pd.to_datetime([start_date, end_date])
        lon_bounds = [-180, 180]
        lat_bounds = [-20, 15]

    # Read precipitation data
    dsr = xr.open_dataset(rain_file)
    ntimes = dsr.sizes['time']
    lon_r = dsr['lon']
    lat_r = dsr['lat']

    # Read mask data
    dsm = xr.open_dataset(mask_file)
    # Replace lat/lon
    dsm['lon'] = lon_r
    dsm['lat'] = lat_r

    # Combine DataSets
    ds = xr.merge([dsr, dsm], compat='no_conflicts')
    # Subset lat/lon to specified region
    lon_sub = ds['lon'].sel(lon=slice(lon_bounds[0], lon_bounds[1]))
    lat_sub = ds['lat'].sel(lat=slice(lat_bounds[0], lat_bounds[1]))
    # Get number of grids
    ny = lat_sub.sizes['lat']
    nx = lon_sub.sizes['lon']
    ngrids = nx * ny

    # Sum total precipitation over space, then divide by total area
    totprecip = ds['precipitation'].sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    ).sum(dim=('lat', 'lon')) / ngrids
    # Sum MCS precipitation over space, use cloudtracknumber > 0 as mask, then divide by total area
    mcsprecip = ds['precipitation'].where(ds['mcs_mask'] > 0).sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    ).sum(dim=('lat', 'lon')) / ngrids

    # Sum mask counts over space, then divide by total area to get fractional cover
    ccscloudfrac = (ds['cloud_mask'] > 0).sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    ).sum(dim=('lat', 'lon')) / ngrids
    mcscloudfrac = (ds['mcs_mask'] > 0).sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    ).sum(dim=('lat', 'lon')) / ngrids
    mcspcpfrac = (ds['mcs_mask'] > 0).sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    ).sum(dim=('lat', 'lon')) / ngrids

    # Interpolate time series to the standard time
    # out_totprecip = totprecip.interp(time=all_datetimes, method='linear')
    # out_mcsprecip = mcsprecip.interp(time=all_datetimes, method='linear')
    # out_mcscloudfrac = mcscloudfrac.interp(time=all_datetimes, method='linear')
    # out_ccscloudfrac = ccscloudfrac.interp(time=all_datetimes, method='linear')
    # out_mcspcpfrac = mcspcpfrac.interp(time=all_datetimes, method='linear')

    #-------------------------------------------------------------------------
    # Write output time series file
    #-------------------------------------------------------------------------
    var_dict = {
        'precipitation': (['time'], totprecip.data),
        'mcs_precipitation': (['time'], mcsprecip.data),
        'mcs_precipitation_frac': (['time'], mcspcpfrac.data),
        'mcs_cloud_frac': (['time'], mcscloudfrac.data),
        'ccs_cloud_frac': (['time'], ccscloudfrac.data),
    }
    coord_dict = {
        'time': (['time'], ds['time'].data),
    }
    gattr_dict = {
        'title': f'Mean precipitation time series',
        'contact':'Zhe Feng, zhe.feng@pnnl.gov',
        'lon_bounds': lon_bounds,
        'lat_bounds': lat_bounds,
        'tracker': tracker,
        'created_on':time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    fillvalue = np.nan
    # Set encoding/compression for all variables
    comp = dict(zlib=True, _FillValue=fillvalue)
    encoding = {var: comp for var in dsout.data_vars}
    dsout.to_netcdf(path=out_filename_timeseries, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'Time series saved: {out_filename_timeseries}')

    # import pdb; pdb.set_trace()