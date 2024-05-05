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

    pcp_dir = f'/pscratch/sd/f/feng045/DYAMOND/OLR_Precipitation_combined_regrid/'
    env_dir = f'/pscratch/sd/f/feng045/DYAMOND/{PHASE}/{runname}/envs_regrid/'
    mask_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask_regrid/{PHASE}/{tracker}/'
    lm_dir = f'/pscratch/sd/f/feng045/DYAMOND/maps/'
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_stats/{PHASE}/{tracker}/'

    pcp_file = f"{pcp_dir}olr_pcp_{PHASE}_{runname}.nc"
    mask_file = f"{mask_dir}mcs_mask_{PHASE}_{runname}.nc"
    env_file = f"{env_dir}{PHASE}_{runname}_{varname}.nc"
    lm_file = f"{lm_dir}era5_landmask.nc"
    out_filename = f'{out_dir}mcs_{varname}_{PHASE}_{runname}.nc'
    out_filename_quartile = f'{out_dir}mcs_{varname}_{PHASE}_{runname}_quartile.nc'

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
    ds_lm = xr.open_dataset(lm_file).sel(lat=slice(min(lat_bounds), max(lat_bounds))).squeeze()
    ds_lm = ds_lm.drop_vars(['utc_date'])

    # import pdb; pdb.set_trace()

    # Convert 'noleap'/'365_day' calendar time to datetime to DatetimeIndex (e.g., SCREAM)
    ds_p = xr.open_dataset(pcp_file)
    ds_p_calendar = ds_p['time'].encoding.get('calendar')
    if (ds_p_calendar == '365_day') | (ds_p_calendar == 'noleap'):
        ds_p = convert_calendar(ds_p, 'noleap')
    ds_m = xr.open_dataset(mask_file)
    ds_m_calendar = ds_m['time'].encoding.get('calendar')
    if (ds_m_calendar == '365_day') | (ds_m_calendar == 'noleap'):        
        ds_m = convert_calendar(ds_m, 'noleap')
    ds_e = xr.open_dataset(env_file)
    ds_e_calendar = ds_e['time'].encoding.get('calendar')
    if (ds_e_calendar == '365_day') | (ds_e_calendar == 'noleap'):        
        ds_e = convert_calendar(ds_e, 'noleap')

    # Read precipitation file
    ds_p = ds_p.sel(
        time=slice(datetime_range[0], datetime_range[1]), lat=slice(min(lat_bounds), max(lat_bounds)))
    time_p = ds_p.time.dt.floor('h')

    # Read MCS mask file
    ds_m = ds_m.sel(
        time=slice(datetime_range[0], datetime_range[1]), lat=slice(min(lat_bounds), max(lat_bounds)))
    time_m = ds_m.time.dt.floor('h')

    # Read environment file
    ds_e = ds_e.sel(
        time=slice(datetime_range[0], datetime_range[1]), lat=slice(min(lat_bounds), max(lat_bounds)))
    time_e = ds_e.time.dt.floor('h')

    # Find the common times among the three DataSets
    common_times = np.intersect1d(np.intersect1d(time_p, time_m), time_e)
    # import pdb; pdb.set_trace()

    # TODO: For testing
    # common_times = common_times[0:24]

    # Subset each DataSet to the common times
    ds_p_common = ds_p.sel(time=common_times)
    ds_m_common = ds_m.sel(time=common_times)
    ds_e_common = ds_e.sel(time=common_times)
    # import pdb; pdb.set_trace()

    # Combine the DataSets into a new DataSet
    ds = xr.merge([ds_lm, ds_p_common, ds_m_common, ds_e_common], combine_attrs='drop_conflicts')
    print(f'Finished combining DataSets.')
    # Subset specified time period
    # ds = ds.sel(time=slice(datetime_range[0], datetime_range[1]))

    intqv = ds['intqv']
    # Make bins for intqv
    bins_intqv = np.arange(5, 95, 2)
    # Bin center values
    bins_intqv_c = bins_intqv[:-1] + (bins_intqv[1:] - bins_intqv[:-1])/2

    print(f'Masking land vs. ocean precipitation ...')
    # Ocean
    oceanfrac_thresh = 0.0
    totpcp_o = ds['precipitation'].where((ds['landseamask'] == oceanfrac_thresh))
    mcspcp_o = ds['precipitation'].where((ds['mcs_mask'] > 0) & (ds['landseamask'] == oceanfrac_thresh))

    # Land
    landfrac_thresh = 0.9
    totpcp_l = ds['precipitation'].where((ds['landseamask'] > landfrac_thresh))
    mcspcp_l = ds['precipitation'].where((ds['mcs_mask'] > 0) & (ds['landseamask'] > landfrac_thresh))

    # Group precipitation by environment and compute mean
    print(f'Running groupby operations ...')
    totpcp_groupby_o = totpcp_o.groupby_bins(intqv, bins=bins_intqv)
    totpcp_groupby_l = totpcp_l.groupby_bins(intqv, bins=bins_intqv)

    mcspcp_groupby_o = mcspcp_o.groupby_bins(intqv, bins=bins_intqv)
    mcspcp_groupby_l = mcspcp_l.groupby_bins(intqv, bins=bins_intqv)

    # Mean values
    totpcp_intqv_o = totpcp_groupby_o.mean(keep_attrs=True)
    totpcp_intqv_l = totpcp_groupby_l.mean(keep_attrs=True)

    mcspcp_intqv_o = mcspcp_groupby_o.mean(keep_attrs=True)
    mcspcp_intqv_l = mcspcp_groupby_l.mean(keep_attrs=True)

    # Number of samples
    totpcp_ns_o = totpcp_groupby_o.count()
    mcspcp_ns_o = mcspcp_groupby_o.count()

    totpcp_ns_l = totpcp_groupby_l.count()
    mcspcp_ns_l = totpcp_groupby_l.count()

    # Define xarray output dataset
    print('Writing output to netCDF file ...')
    var_dict = {
        'total_land': (['bins'], totpcp_intqv_l.data, totpcp_intqv_l.attrs),
        'mcs_land': (['bins'], mcspcp_intqv_l.data, mcspcp_intqv_l.attrs),
        'total_ocean': (['bins'], totpcp_intqv_o.data, totpcp_intqv_o.attrs),
        'mcs_ocean': (['bins'], mcspcp_intqv_o.data, mcspcp_intqv_o.attrs),

        'total_land_ns': (['bins'], totpcp_ns_l.data),
        'mcs_land_ns': (['bins'], mcspcp_ns_l.data),
        'total_ocean_ns': (['bins'], totpcp_ns_o.data),
        'mcs_ocean_ns': (['bins'], mcspcp_ns_o.data),
    }
    coord_dict = {'bins': (['bins'], bins_intqv_c)}
    gattr_dict = {
        'title': 'Precipitation by environments',
        'lon_bounds': lon_bounds,
        'lat_bounds': lat_bounds,
        'landfrac_thresh': landfrac_thresh,
        'oceanfrac_thresh': oceanfrac_thresh,
        'contact':'Zhe Feng, zhe.feng@pnnl.gov',
        'created_on':time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
    # Define variable attributes
    # dsout.bins.attrs['long_name'] = f"{intqv.attrs['long_name']} bins"
    # dsout.bins.attrs['units'] = intqv.attrs['units']
    dsout.bins.attrs['long_name'] = f"Total column water vapor bins"
    dsout.bins.attrs['units'] = f"kg m**-2"

    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float')
    encoding = {var: comp for var in dsout.data_vars}
    # Write to file
    dsout.to_netcdf(path=out_filename, mode='w', format='NETCDF4', encoding=encoding)
    print('Output saved as: ', out_filename)



    # For OBS, compute interquatile range and save to a separate file
    if 'OBS' in runname:
        print(f'Computing interquartile range ...')
        # Compute interquartile range
        mcspcp25_intqv_o = mcspcp_groupby_o.quantile(0.25, keep_attrs=True)
        mcspcp75_intqv_o = mcspcp_groupby_o.quantile(0.75, keep_attrs=True)
        mcspcp25_intqv_l = mcspcp_groupby_l.quantile(0.25, keep_attrs=True)
        mcspcp75_intqv_l = mcspcp_groupby_l.quantile(0.75, keep_attrs=True)

        # # Standard deviation
        # totpcp_intqv_std_o = totpcp_groupby_o.std()
        # mcspcp_intqv_std_o = mcspcp_groupby_o.std()

        # totpcp_intqv_std_l = totpcp_groupby_l.std()
        # mcspcp_intqv_std_l = totpcp_groupby_l.std()

        print('Writing interquartile output to netCDF file ...')
        var_dict = {
            # Interquartile range values
            'mcs_ocean_25': (['bins'], mcspcp25_intqv_o.data, mcspcp25_intqv_o.attrs),
            'mcs_ocean_75': (['bins'], mcspcp75_intqv_o.data, mcspcp75_intqv_o.attrs),
            'mcs_land_25': (['bins'], mcspcp25_intqv_l.data, mcspcp25_intqv_l.attrs),
            'mcs_land_75': (['bins'], mcspcp75_intqv_l.data, mcspcp75_intqv_l.attrs),

            # 'total_land_std': (['bins'], totpcp_intqv_std_l.data),
            # 'mcs_land_std': (['bins'], mcspcp_intqv_std_l.data),
            # 'total_ocean_std': (['bins'], totpcp_intqv_std_o.data),
            # 'mcs_ocean_std': (['bins'], mcspcp_intqv_std_o.data),
        }
        coord_dict = {'bins': (['bins'], bins_intqv_c)}
        gattr_dict = {
            'title': 'Precipitation by environments interquartile values',
            'lon_bounds': lon_bounds,
            'lat_bounds': lat_bounds,
            'landfrac_thresh': landfrac_thresh,
            'oceanfrac_thresh': oceanfrac_thresh,
            'contact':'Zhe Feng, zhe.feng@pnnl.gov',
            'created_on':time.ctime(time.time()),
        }
        dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
        # Define variable attributes
        dsout.bins.attrs['long_name'] = f"Total column water vapor bins"
        dsout.bins.attrs['units'] = f"kg m**-2"

        # Set encoding/compression for all variables
        comp = dict(zlib=True, dtype='float')
        encoding = {var: comp for var in dsout.data_vars}
        # Write to file
        dsout.to_netcdf(path=out_filename_quartile, mode='w', format='NETCDF4', encoding=encoding)
        print('Output interquartile saved as: ', out_filename_quartile)