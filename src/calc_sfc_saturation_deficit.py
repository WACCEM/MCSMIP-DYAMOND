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

def saturation_vapor_pressure(temperature):
    r"""Calculate the saturation water vapor (partial) pressure.

    Parameters
    ----------
    temperature : array-like
        Air temperature [Kelvin]

    Returns
    -------
    array-like
        Saturation water vapor (partial) pressure

    Notes
    -----
    Taken from MetPy function:
    https://github.com/Unidata/MetPy/blob/34bfda1deaead3fed9070f3a766f7d842373c6d9/src/metpy/calc/thermo.py#L1285

    The formula used is that from [Bolton1980]_ for T in degrees Celsius:

    .. math:: 6.112 e^\frac{17.67T}{T + 243.5}

    """
    # Converted from original in terms of C to use kelvin.
    return 611.2 * np.exp(
        17.67 * (temperature - 273.15) / (temperature - 29.65)
    )

def mixing_ratio(partial_press, total_press, molecular_weight_ratio=0.622):
    r"""Calculate the mixing ratio of a gas.

    This calculates mixing ratio given its partial pressure and the total pressure of
    the air. There are no required units for the input arrays, other than that
    they have the same units.

    Parameters
    ----------
    partial_press : array-like
        Partial pressure of the constituent gas

    total_press : array-like
        Total air pressure

    molecular_weight_ratio : float, optional
        The ratio of the molecular weight of the constituent gas to that assumed
        for air. Defaults to the ratio for water vapor to dry air
        (:math:`\epsilon\approx0.622`).

    Returns
    -------
    array-like
        The (mass) mixing ratio, dimensionless (e.g. Kg/Kg or g/g)

    Notes
    -----
    Taken from MetPy function:
    https://github.com/Unidata/MetPy/blob/34bfda1deaead3fed9070f3a766f7d842373c6d9/src/metpy/calc/thermo.py#L1418

    This function is a straightforward implementation of the equation given in many places,
    such as [Hobbs1977]_ pg.73:

    .. math:: r = \epsilon \frac{e}{p - e}

    .. versionchanged:: 1.0
       Renamed ``part_press``, ``tot_press`` parameters to ``partial_press``, ``total_press``

    """
    return molecular_weight_ratio * partial_press / (total_press - partial_press)

def saturation_mixing_ratio(total_press, temperature):
    r"""Calculate the saturation mixing ratio of water vapor.

    This calculation is given total atmospheric pressure and air temperature.

    Parameters
    ----------
    total_press: array-like
        Total atmospheric pressure [Pa]

    temperature: array-like
        Air temperature [K]

    Returns
    -------
    array-like
        Saturation mixing ratio, dimensionless

    Notes
    -----
    Taken from MetPy function:
    https://github.com/Unidata/MetPy/blob/34bfda1deaead3fed9070f3a766f7d842373c6d9/src/metpy/calc/thermo.py#L1474

    This function is a straightforward implementation of the equation given in many places,
    such as [Hobbs1977]_ pg.73:

    .. math:: r_s = \epsilon \frac{e_s}{p - e_s}

    .. versionchanged:: 1.0
       Renamed ``tot_press`` parameter to ``total_press``

    """
    return mixing_ratio(saturation_vapor_pressure(temperature), total_press)



if __name__ == "__main__":

    PHASE = sys.argv[1]
    runname = sys.argv[2]

    # Inputs
    env_dir = f'/pscratch/sd/f/feng045/DYAMOND/Environments_0.25deg/'
    file_t2m = f"{env_dir}{PHASE}_{runname}_t2m.nc"
    file_q2m = f"{env_dir}{PHASE}_{runname}_q2m.nc"
    file_ps = f"{env_dir}{PHASE}_{runname}_ps.nc"

    # Outputs 
    out_dir = env_dir
    out_filename = f'{out_dir}{PHASE}_{runname}_q2sd.nc'

    # Start time for encoding
    if PHASE == 'Summer':
        start_datetime = '2016-08-01 00:00:00'
    elif PHASE == 'Winter':
        start_datetime = '2020-01-20 00:00:00' 


    # Convert 'noleap'/'365_day' calendar time to datetime to DatetimeIndex (e.g., SCREAM)
    ds_t2m = xr.open_dataset(file_t2m)
    ds_t2m_calendar = ds_t2m['time'].encoding.get('calendar')
    if (ds_t2m_calendar == '365_day') | (ds_t2m_calendar == 'noleap'):        
        ds_t2m = convert_calendar(ds_t2m, 'noleap')
    ds_q2m = xr.open_dataset(file_q2m)
    ds_q2m_calendar = ds_q2m['time'].encoding.get('calendar')
    if (ds_q2m_calendar == '365_day') | (ds_q2m_calendar == 'noleap'):        
        ds_q2m = convert_calendar(ds_q2m, 'noleap')
    ds_ps = xr.open_dataset(file_ps)
    ds_ps_calendar = ds_ps['time'].encoding.get('calendar')
    if (ds_ps_calendar == '365_day') | (ds_ps_calendar == 'noleap'):        
        ds_ps = convert_calendar(ds_ps, 'noleap')

    # Read t2m file
    time_t2m = ds_t2m.time.dt.floor('h')
    # Read q2m file
    time_q2m = ds_q2m.time.dt.floor('h')
    # Read ps file
    time_ps = ds_ps.time.dt.floor('h')

    # Find the common times among the DataSets
    common_times = np.intersect1d(np.intersect1d(time_t2m, time_q2m), time_ps)

    # TODO: For testing
    # common_times = common_times[100:124]

    # Subset each DataSet to the common times
    ds_t2m_common = ds_t2m.sel(time=common_times)
    ds_q2m_common = ds_q2m.sel(time=common_times)
    ds_ps_common = ds_ps.sel(time=common_times)

    # Combine the DataSets into a new DataSet
    ds = xr.merge([ds_t2m_common, ds_q2m_common, ds_ps_common], combine_attrs='drop_conflicts')
    print(f'Finished combining DataSets.')

    lon = ds['lon']
    lat = ds['lat']
    # Change time coordinate encoding
    ds['time'].encoding['units'] = f'hours since {start_datetime}'

    # Compute saturation mixing ratio
    q2sat_mr = saturation_mixing_ratio(ds['ps'], ds['t2m'])
    # Convert mixing ratio to specific humidity
    q2sat = q2sat_mr / (1 + q2sat_mr)
    # Saturation deficit, convert unit to [g/kg]
    q2sd = 1000 * (q2sat - ds['q2m'])

    # Add attributes
    q2sd.attrs['long_name'] = 'Surface saturation deficit'
    q2sd.attrs['units'] = 'g kg^-1'


    #-------------------------------------------------------------------------
    # Write output map file
    #-------------------------------------------------------------------------
    # Define xarray output dataset
    print('Writing output to netCDF file ...')
    coord_names = ['time', 'lat', 'lon']
    var_dict = {
        'q2sd': (coord_names, q2sd.data, q2sd.attrs),
    }
    coord_dict = {
        'time': (['time'], ds['time'].data),
        'lat': (['lat'], lat.data, lat.attrs),
        'lon': (['lon'], lon.data, lon.attrs),
    }
    gattr_dict = {
        'title': f'Surface saturation deficit',
        'contact':'Zhe Feng, zhe.feng@pnnl.gov',
        'created_on':time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
    fillvalue = np.nan
    # Set encoding/compression for all variables
    comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    dsout.to_netcdf(path=out_filename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'Map saved: {out_filename}')