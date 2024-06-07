"""
Calculate global mean total, MCS precipitation time series and save output to a netCDF file.
"""
import numpy as np
import sys, os
import xarray as xr
import pandas as pd
import time
import cftime

def convert_calendar(ds, target_calendar, freq='1h', time_coord_name='time'):
    """
    Convert calendar encoding for an Xarray DataSet

    Args:
        ds: Xarray DataSet
            Input Xarray DataSet.
        target_calendar: string
            Calendar to convert the encoding of the DataSet to.
        freq: string, default='1h'
            Frequency of the time variable
        time_coord_name: string, default='time'
            Name of the time coordinate in the DataSet

    Returns:
        ds: Xarray DataSet
            Output Xarray DataSet
    """
    # Get number of times from the DataSet
    ntimes = ds.sizes[time_coord_name]
    # Make a new DatetimeIndex for specific frequency and calendar
    time_DatetimeIndex = xr.cftime_range(start=ds[time_coord_name].values[0], periods=ntimes, freq=freq, calendar=target_calendar).to_datetimeindex()
    # Convert DatetimeIndex to DataArray, then replace the time coordinate in the DataSet
    time_mcs_mask = xr.DataArray(time_DatetimeIndex, coords={time_coord_name: time_DatetimeIndex}, dims=time_coord_name)
    ds[time_coord_name] = time_mcs_mask
    return ds


if __name__ == "__main__":

    PHASE = sys.argv[1]
    runname = sys.argv[2]
    # start_date = sys.argv[3]
    # end_date = sys.argv[4]

    # Inputs
    tracker = 'PyFLEXTRKR'
    pcp_dir = f'/pscratch/sd/f/feng045/DYAMOND/OLR_Precipitation_combined/'
    mask_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask/{PHASE}/{tracker}/'
    lm_dir = '/pscratch/sd/f/feng045/DYAMOND/maps/'
    rain_file = f'{pcp_dir}olr_pcp_{PHASE}_{runname}.nc'
    mask_file = f'{mask_dir}mcs_mask_{PHASE}_{runname}.nc'
    lm_file = f'{lm_dir}IMERG_landmask_180W-180E_60S-60N.nc'
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

    # Ocean landseamask range
    ocean_range = [99, 100]

    # Read landmask data
    dslm = xr.open_dataset(lm_file)
    landseamask = dslm['landseamask'].squeeze()

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

    # Convert 'noleap'/'365_day' calendar time to datetime to DatetimeIndex (e.g., SCREAM)
    if (dsr['time'].encoding.get('calendar') == '365_day'):
        dsr = convert_calendar(dsr, 'noleap')
    if (dsm['time'].encoding.get('calendar') == '365_day'):        
        dsm = convert_calendar(dsm, 'noleap')

    # Combine DataSets
    ds = xr.merge([dsr, dsm], compat='no_conflicts')

    # Subset lat/lon to specified region
    lon_sub = ds['lon'].sel(lon=slice(lon_bounds[0], lon_bounds[1]))
    lat_sub = ds['lat'].sel(lat=slice(lat_bounds[0], lat_bounds[1]))
    # Get number of grids
    ny = lat_sub.sizes['lat']
    nx = lon_sub.sizes['lon']
    ngrids = nx * ny

    # Make a mask for ocean
    mask_ocean = (landseamask >= min(ocean_range)) & (landseamask <= max(ocean_range))
    # Count the number of grids over ocean
    ngrids_o = mask_ocean.sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    ).sum().values

    # Oceanic precipitation
    totprecip_o = ds['precipitation'].where(mask_ocean).sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    ).sum(dim=('lat', 'lon')) / ngrids_o
    # mcsprecip_o = ds['precipitation'].where((ds['mcs_mask'] > 0) & (mask_ocean)).sel(
    #     lon=slice(lon_bounds[0], lon_bounds[1]), 
    #     lat=slice(lat_bounds[0], lat_bounds[1]),
    # ).sum(dim=('lat', 'lon')) / ngrids_o

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
        'precipitation_ocean': (['time'], totprecip_o.data),
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