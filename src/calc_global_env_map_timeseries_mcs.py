import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import time
import cftime

# def convert_calendar(ds, target_calendar):
#     # Convert the time coordinate to a specific calendar
#     time_var = ds['time']
#     num_dates = cftime.date2num(time_var.values, time_var.encoding.get('units', None), calendar=target_calendar)
#     converted_dates = cftime.num2date(num_dates, time_var.encoding.get('units', None), calendar=target_calendar)
    
#     # Update the dataset
#     ds['time'] = ('time', converted_dates)
#     ds['time'].attrs['calendar'] = target_calendar
#     return ds

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
    tracker = sys.argv[3]
    varname = sys.argv[4]
    start_date = sys.argv[5]
    end_date = sys.argv[6]

    # Landmask file
    lm_dir = '/pscratch/sd/f/feng045/DYAMOND/maps/'
    lm_file = f"{lm_dir}era5_landmask.nc"
    ocean_range = [0, 0.01]
    # if ('OBS' in runname):
    #     lm_file = f'{lm_dir}era5_landmask.nc'
    #     ocean_range = [0, 0.01]
    # else:
    #     lm_file = f'{lm_dir}IMERG_landmask_180W-180E_60S-60N.nc'
    #     ocean_range = [99, 100]

    # Inputs
    env_dir = f'/pscratch/sd/f/feng045/DYAMOND/Environments_0.25deg/'
    env_file = f"{env_dir}{PHASE}_{runname}_{varname}.nc"
    mask_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask_regrid/{PHASE}/{tracker}/'
    mask_file = f"{mask_dir}mcs_mask_{PHASE}_{runname}.nc"

    # Outputs 
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/{PHASE}/{runname}/envs/'
    out_filename_timeseries = f'{out_dir}{PHASE}_{runname}_{varname}_timeseries.nc'
    out_filename_map = f'{out_dir}{PHASE}_{runname}_{varname}_map.nc'

    # Specify regions and time periods for averaging
    if PHASE == 'Summer':
        datetime_range = pd.to_datetime([start_date, end_date])
        lon_bounds = [-180, 180]
        lat_bounds = [-15, 30]
    if PHASE == 'Winter':
        datetime_range = pd.to_datetime([start_date, end_date])
        lon_bounds = [-180, 180]
        lat_bounds = [-20, 15]

    # Read landmask file
    ds_lm = xr.open_dataset(lm_file).squeeze()
    ds_lm = ds_lm.drop_vars(['utc_date'])
    landseamask = ds_lm['landseamask']

    # import pdb; pdb.set_trace()

    # Convert 'noleap'/'365_day' calendar time to datetime to DatetimeIndex (e.g., SCREAM)
    # ds_p = xr.open_dataset(pcp_file)
    # ds_p_calendar = ds_p['time'].encoding.get('calendar')
    # if (ds_p_calendar == '365_day') | (ds_p_calendar == 'noleap'):
    #     ds_p = convert_calendar(ds_p, 'noleap')
    ds_m = xr.open_dataset(mask_file)
    ds_m_calendar = ds_m['time'].encoding.get('calendar')
    if (ds_m_calendar == '365_day') | (ds_m_calendar == 'noleap'):        
        ds_m = convert_calendar(ds_m, 'noleap')
    ds_e = xr.open_dataset(env_file)
    ds_e_calendar = ds_e['time'].encoding.get('calendar')
    if (ds_e_calendar == '365_day') | (ds_e_calendar == 'noleap'):        
        ds_e = convert_calendar(ds_e, 'noleap')

    # Read MCS mask file
    # ds_m = ds_m.sel(
    #     time=slice(datetime_range[0], datetime_range[1]), lat=slice(min(lat_bounds), max(lat_bounds)))
    time_m = ds_m.time.dt.floor('h')

    # Read environment file
    # ds_e = ds_e.sel(
    #     time=slice(datetime_range[0], datetime_range[1]), lat=slice(min(lat_bounds), max(lat_bounds)))
    time_e = ds_e.time.dt.floor('h')

    # Find the common times among the DataSets
    common_times = np.intersect1d(time_e, time_m)

    # TODO: For testing
    # common_times = common_times[0:24]

    # Subset each DataSet to the common times
    ds_m_common = ds_m.sel(time=common_times)
    ds_e_common = ds_e.sel(time=common_times)

    # Combine the DataSets into a new DataSet
    ds = xr.merge([ds_lm, ds_m_common, ds_e_common], combine_attrs='drop_conflicts')
    print(f'Finished combining DataSets.')

    lon = ds['lon']
    lat = ds['lat']
    # Get number of grids
    ny = lat.sizes['lat']
    nx = lon.sizes['lon']
    ngrids = nx * ny
    # Get number of times
    ntimes = ds.sizes['time']  # all times
    ntimes_s = ds['time'].sel(time=slice(start_date, end_date)).sizes['time']  # within the subset period

    VAR = ds[varname]
    VAR_units = ds[varname].units
    mcs_mask = ds['mcs_mask']
    cloud_mask = ds['cloud_mask']

    # Convert units
    if (varname == 'lhx') & (VAR_units == 'm^-2 s^-1 kg'):
        Lv = 2.26e6  # Latent heat of vaporization [J kg^-1]
        VAR_units = 'W m-2'
        # Get variable attributes
        VAR_attrs = VAR.attrs
        # Convert latent heat flux unit to [W m^-2]
        VAR = (VAR * Lv).assign_attrs(VAR_attrs)
        # Update units
        VAR.attrs['units'] = VAR_units
        # Update Dataset
        ds[varname] = VAR

    # Compute CCS/MCS frequency
    ccs_freq = 100 * (cloud_mask > 0).sel(time=slice(start_date, end_date)).sum(dim='time', keep_attrs=True) / ntimes_s
    ccs_freq.attrs['long_name'] = 'CCS frequency'
    ccs_freq.attrs['units'] = '%'

    mcs_freq = 100 * (mcs_mask > 0).sel(time=slice(start_date, end_date)).sum(dim='time', keep_attrs=True) / ntimes_s
    mcs_freq.attrs['long_name'] = 'MCS frequency'
    mcs_freq.attrs['units'] = '%'
    # import pdb; pdb.set_trace()

    # Compute time sum then divide by ntimes to get mean
    VAR_avg_map = VAR.sel(time=slice(start_date, end_date)).sum(dim='time', keep_attrs=True) / ntimes_s
    VAR_ccs_map = VAR.where(cloud_mask > 0).sel(time=slice(start_date, end_date)).sum(dim='time', keep_attrs=True) / ntimes_s
    VAR_mcs_map = VAR.where(mcs_mask > 0).sel(time=slice(start_date, end_date)).sum(dim='time', keep_attrs=True) / ntimes_s

    # Compute conditional mean
    VAR_ccs_condmap = VAR.where(cloud_mask > 0).sel(time=slice(start_date, end_date)).mean(dim='time', keep_attrs=True)
    VAR_mcs_condmap = VAR.where(mcs_mask > 0).sel(time=slice(start_date, end_date)).mean(dim='time', keep_attrs=True)
    # Update DataArray attribute
    VAR_ccs_condmap.attrs['long_name'] = VAR_ccs_condmap.attrs.get('long_name', f"{varname}") + " conditional mean"
    VAR_mcs_condmap.attrs['long_name'] = VAR_mcs_condmap.attrs.get('long_name', f"{varname}") + " conditional mean"
    # import pdb; pdb.set_trace()

    #-------------------------------------------------------------------------
    # Time series calculation
    #-------------------------------------------------------------------------
    # Subset lat band for averaging in space
    ds_s = ds.sel(
        lat=slice(min(lat_bounds), max(lat_bounds))
    )
    # Get number of grids
    ny_s = ds_s.sizes['lat']
    nx_s = ds_s.sizes['lon']
    ngrids_s = nx_s * ny_s
    VAR_s = ds_s[varname]
    landseamask_s = ds_s['landseamask']
    # Sum over space, then divide by the total area
    VAR_avg_timeseries = VAR_s.sum(dim=('lon', 'lat'), keep_attrs=True) / ngrids_s
    # Make a mask for ocean
    mask_ocean = (landseamask_s >= min(ocean_range)) & (landseamask_s <= max(ocean_range))
    ngrids_o_s = np.count_nonzero(mask_ocean)
    # Sum over space, then divide by the ocean area
    VAR_o_avg_timeseries = VAR_s.where(mask_ocean).sum(dim=('lon', 'lat'), keep_attrs=True) / ngrids_o_s
    VAR_o_ccs_avg_timeseries = VAR_s.where((mask_ocean) & (cloud_mask > 0)).sum(dim=('lon', 'lat'), keep_attrs=True) / ngrids_o_s
    VAR_o_mcs_avg_timeseries = VAR_s.where((mask_ocean) & (mcs_mask > 0)).sum(dim=('lon', 'lat'), keep_attrs=True) / ngrids_o_s
    # Conditional mean
    VAR_o_ccs_condavg_timeseries = VAR_s.where((mask_ocean) & (cloud_mask > 0)).mean(dim=('lon', 'lat'), keep_attrs=True)
    VAR_o_mcs_condavg_timeseries = VAR_s.where((mask_ocean) & (mcs_mask > 0)).mean(dim=('lon', 'lat'), keep_attrs=True)
    # Update DataArray attribute
    VAR_o_ccs_condavg_timeseries.attrs['long_name'] = VAR_o_ccs_condavg_timeseries.attrs.get('long_name', f"{varname}") + " conditional mean"
    VAR_o_mcs_condavg_timeseries.attrs['long_name'] = VAR_o_mcs_condavg_timeseries.attrs.get('long_name', f"{varname}") + " conditional mean"

    # import matplotlib.pyplot as plt
    # VAR_o_avg_timeseries.plot(label='All', color='k')
    # # # VAR_o_ccs_avg_timeseries.plot(label='CCS', color='g')
    # # # VAR_o_mcs_avg_timeseries.plot(label='MCS', color='r')
    # # (100 * VAR_o_ccs_avg_timeseries / VAR_o_avg_timeseries).plot(label='CCS', color='g')
    # # (100 * VAR_o_mcs_avg_timeseries / VAR_o_avg_timeseries).plot(label='MCS', color='r')
    # VAR_o_ccs_condavg_timeseries.plot(label='CCS', color='g')
    # VAR_o_mcs_condavg_timeseries.plot(label='MCS', color='r')
    # plt.legend()
    # plt.show()
    # import pdb; pdb.set_trace()

    #-------------------------------------------------------------------------
    # Write output map file
    #-------------------------------------------------------------------------
    # Define xarray output dataset
    print('Writing output to netCDF file ...')
    coord_names = ['lat', 'lon']
    var_dict = {
        'landseamask': (coord_names, landseamask.data, landseamask.attrs),
        f'{varname}': (coord_names, VAR_avg_map.data, VAR_avg_map.attrs),
        f'{varname}_ccs': (coord_names, VAR_ccs_map.data, VAR_ccs_map.attrs),
        f'{varname}_mcs': (coord_names, VAR_mcs_map.data, VAR_mcs_map.attrs),
        f'{varname}_ccs_cond': (coord_names, VAR_ccs_condmap.data, VAR_ccs_condmap.attrs),
        f'{varname}_mcs_cond': (coord_names, VAR_mcs_condmap.data, VAR_mcs_condmap.attrs),
        'ccs_freq': (coord_names, ccs_freq.data, ccs_freq.attrs),
        'mcs_freq': (coord_names, mcs_freq.data, mcs_freq.attrs),
        'ntimes': xr.DataArray(ntimes_s, attrs={"long_name": "Number of times in the averaging", "units": "counts"}),
    }
    coord_dict = {
        'lat': (['lat'], lat.data, lat.attrs),
        'lon': (['lon'], lon.data, lon.attrs),
    }
    gattr_dict = {
        'title': f'Time sum {varname} map',
        'ntimes': ntimes_s,
        'contact':'Zhe Feng, zhe.feng@pnnl.gov',
        'start_date': datetime_range[0].strftime('%Y-%m-%dT%H'),
        'end_date': datetime_range[1].strftime('%Y-%m-%dT%H'),
        'created_on':time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
    fillvalue = np.nan
    # Set encoding/compression for all variables
    comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    dsout.to_netcdf(path=out_filename_map, mode='w', format='NETCDF4', encoding=encoding)
    print(f'Map saved: {out_filename_map}')


    #-------------------------------------------------------------------------
    # Write output time series file
    #-------------------------------------------------------------------------
    var_dict = {
        f'{varname}': (['time'], VAR_avg_timeseries.data, VAR.attrs),
        f'{varname}_ocean': (['time'], VAR_o_avg_timeseries.data, VAR.attrs),
        f'{varname}_ocean_ccs': (['time'], VAR_o_ccs_avg_timeseries.data, VAR.attrs),
        f'{varname}_ocean_mcs': (['time'], VAR_o_mcs_avg_timeseries.data, VAR.attrs),
        f'{varname}_ocean_ccs_cond': (['time'], VAR_o_ccs_condavg_timeseries.data, VAR_o_ccs_condavg_timeseries.attrs),
        f'{varname}_ocean_mcs_cond': (['time'], VAR_o_mcs_condavg_timeseries.data, VAR_o_mcs_condavg_timeseries.attrs),
    }
    coord_dict = {
        'time': (['time'], ds_s['time'].data),
    }
    gattr_dict = {
        'title': f'Mean {varname} time series',
        'lon_bounds': lon_bounds,
        'lat_bounds': lat_bounds,
        'ngrids': ngrids_s,
        'ngrids_o': ngrids_o_s,
        'contact':'Zhe Feng, zhe.feng@pnnl.gov',
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