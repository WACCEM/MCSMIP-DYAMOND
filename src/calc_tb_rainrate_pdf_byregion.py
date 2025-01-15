"""
Calculate histograms of OLR/Tb, total & MCS precipitation for a region and save output to a netCDF file.
"""
import numpy as np
import sys
import xarray as xr
import pandas as pd
import time

def olr_to_tb(OLR):
    """
    Convert OLR to IR brightness temperature.

    Args:
        OLR: np.array
            Outgoing longwave radiation
    
    Returns:
        tb: np.array
            Brightness temperature
    """
    # Calculate brightness temperature
    # (1984) as given in Yang and Slingo (2001)
    # Tf = tb(a+b*Tb) where a = 1.228 and b = -1.106e-3 K^-1
    # OLR = sigma*Tf^4 
    # where sigma = Stefan-Boltzmann constant = 5.67x10^-8 W m^-2 K^-4
    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8 # W m^-2 K^-4
    tf = (OLR/sigma)**0.25
    tb = (-a + np.sqrt(a**2 + 4*b*tf))/(2*b)
    return tb


if __name__ == "__main__":

    PHASE = sys.argv[1]
    runname = sys.argv[2]
    tracker = sys.argv[3]
    start_date = sys.argv[4]
    end_date = sys.argv[5]

    # Inputs
    tb_varname = 'Tb'
    olr_varname = 'olr'

    # tracker = 'PyFLEXTRKR'
    pcp_dir = f'/pscratch/sd/f/feng045/DYAMOND/OLR_Precipitation_combined/'
    mask_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask/{PHASE}/{tracker}/'
    rain_file = f'{pcp_dir}olr_pcp_{PHASE}_{runname}.nc'
    mask_file = f'{mask_dir}mcs_mask_{PHASE}_{runname}.nc'
    file_landmask = f'/pscratch/sd/f/feng045/DYAMOND/maps/IMERG_landmask_180W-180E_60S-60N.nc'
    # Outputs 
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_stats/{PHASE}/{tracker}/'
    out_filename = f'{out_dir}tb_rainrate_hist_{PHASE}_{runname}.nc'

    # Specify regions and time periods for averaging
    if PHASE == 'Summer':
        # datetime_range = pd.to_datetime([start_date, end_date])
        lon_bounds = [-180, 180]
        lat_bounds = [-15, 30]
    if PHASE == 'Winter':
        # datetime_range = pd.to_datetime([start_date, end_date])
        lon_bounds = [-180, 180]
        lat_bounds = [-20, 15]

    # Ocean vs. Land threshold (%)
    ocean_thresh = 99
    land_thresh = 20

    # Read landmask data
    dslm = xr.open_dataset(file_landmask)
    landseamask = dslm['landseamask']

    # Read precipitation data
    dsr = xr.open_dataset(rain_file)
    ntimes = dsr.sizes['time']
    lon_r = dsr['lon']
    lat_r = dsr['lat']
    print(f'Finished reading: {rain_file}')

    # Read mask data
    dsm = xr.open_dataset(mask_file)
    ntimes_m = dsm.sizes['time']
    # Replace lat/lon
    dsm['lon'] = lon_r
    dsm['lat'] = lat_r
    print(f'Finished reading: {mask_file}')

    # Check time encoding from the precipitation file
    time_encoding = dsr['time'].encoding.get('calendar', None)
    time_encoding_m = dsm['time'].encoding.get('calendar', None)
    # Convert 'noleap'/'365_day' calendar time to datetime to DatetimeIndex (e.g., SCREAM)
    if (time_encoding == '365_day') | (time_encoding == 'noleap'):
        time_DatetimeIndex = xr.cftime_range(start=dsr['time'].values[0], periods=ntimes, freq='1H', calendar='noleap').to_datetimeindex()
        # Convert DatetimeIndex to DataArray, then replace the time coordinate in the DataSet
        time_pcp = xr.DataArray(time_DatetimeIndex, coords={'time': time_DatetimeIndex}, dims='time')
        dsr['time'] = time_pcp

    if (time_encoding == '365_day') | (time_encoding == 'noleap'):
        time_DatetimeIndex = xr.cftime_range(start=dsm['time'].values[0], periods=ntimes_m, freq='1H', calendar='noleap').to_datetimeindex()
        # Convert DatetimeIndex to DataArray, then replace the time coordinate in the DataSet
        time_mcs_mask = xr.DataArray(time_DatetimeIndex, coords={'time': time_DatetimeIndex}, dims='time')
        dsm['time'] = time_mcs_mask

    # Combine DataSets, subset region
    ds = xr.merge([dsr, dsm], compat='no_conflicts').sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
        time=slice(start_date, end_date),
    )
    print(f'Finished combining two datasets.')

    # Red Tb for OBS data, convert OLR to Tb for model data
    if 'OBS' in runname:
        tb = ds[tb_varname]
    else:
        olr = ds[olr_varname]
        tb = olr_to_tb(olr)

    # Set up Tb, rain rate bins
    bins_pcp = np.arange(1, 301, 1)
    bins_tb = np.arange(160, 351, 1)
    bins_olr = np.arange(50, 400, 1)
    # bins_pcp = np.logspace(np.log10(0.01), np.log10(100.0), 100)
    pcp_range = (np.min(bins_pcp), np.max(bins_pcp))
    tb_range = (np.min(bins_tb), np.max(bins_tb))
    olr_range = (np.min(bins_olr), np.max(bins_olr))

    # Compute histograms
    print(f'Computing OLR/Tb histogram ...')
    if 'OBS' not in runname:
        olr_hist, bins = np.histogram(olr, bins=bins_olr, range=olr_range, density=False)
    else:
        olr_hist = np.zeros(len(bins_olr)-1)

    print(f'Computing total precipitation histogram ...')
    tb_hist, bins = np.histogram(tb, bins=bins_tb, range=tb_range, density=False)
    totpcp_hist, bins = np.histogram(ds['precipitation'], bins=bins_pcp, range=pcp_range, density=False)

    # MCS Tb
    mcstb = tb.where(ds['mcs_mask'] > 0)
    mcstb_hist, bins = np.histogram(mcstb, bins=bins_tb, range=tb_range, density=False)

    # MCS precipitation mask
    print(f'Computing MCS precipitation histogram ...')
    mcspcp = ds['precipitation'].where(ds['mcs_mask'] > 0)
    mcspcp_hist, bins = np.histogram(mcspcp, bins=bins_pcp, range=pcp_range, density=False)

    # MCS precipitation (ocean vs. land)
    mcspcp_o = ds['precipitation'].where((ds['mcs_mask'] > 0) & (landseamask >= ocean_thresh))
    mcspcp_l = ds['precipitation'].where((ds['mcs_mask'] > 0) & (landseamask <= land_thresh))
    mcspcp_o_hist, bins = np.histogram(mcspcp_o, bins=bins_pcp, range=pcp_range, density=False)
    mcspcp_l_hist, bins = np.histogram(mcspcp_l, bins=bins_pcp, range=pcp_range, density=False)
    # import pdb; pdb.set_trace()


    #-------------------------------------------------------------------------
    # Write output file
    #-------------------------------------------------------------------------
    # Define xarray output dataset
    print('Writing output to netCDF file ...')
    bins_tb_attrs = {'units': 'K'}
    bins_pcp_attrs = {'units': 'mm h-1'}
    bins_olr_attrs = {'units': 'W m-2'}
    olr_attrs = {'long_name': 'OLR histogram', 'units':'count'}
    tb_attrs = {'long_name': 'Tb histogram', 'units':'count'}
    totpcp_attrs = {'long_name': 'Total precipitation histogram', 'units':'count'}
    mcspcp_attrs = {'long_name': 'MCS precipitation histogram', 'units':'count'}
    mcspcp_o_attrs = {'long_name': 'Ocean MCS precipitation histogram', 'units':'count'}
    mcspcp_l_attrs = {'long_name': 'Land MCS precipitation histogram', 'units':'count'}

    var_dict = {
        'olr': (['bins_olr'], olr_hist, olr_attrs),
        'tb': (['bins_tb'], tb_hist, tb_attrs),
        'total_pcp': (['bins_pcp'], totpcp_hist, totpcp_attrs),
        'mcs_pcp': (['bins_pcp'], mcspcp_hist, mcspcp_attrs),
        'mcs_tb': (['bins_tb'], mcstb_hist, tb_attrs),
        'mcs_pcp_o': (['bins_pcp'], mcspcp_o_hist, mcspcp_o_attrs),
        'mcs_pcp_l': (['bins_pcp'], mcspcp_l_hist, mcspcp_l_attrs),
    }
    coord_dict = {
        'bins_olr': (['bins_olr'], bins_olr[:-1], bins_olr_attrs),
        'bins_tb': (['bins_tb'], bins_tb[:-1], bins_tb_attrs),
        'bins_pcp': (['bins_pcp'], bins_pcp[:-1], bins_pcp_attrs),
    }
    gattr_dict = {
        'title': 'OLR, Tb, precipitation histogram',
        'lon_bounds': lon_bounds,
        'lat_bounds': lat_bounds,
        'start_date': start_date,
        'end_date': end_date,
        'ocean_thresh': ocean_thresh,
        'land_thresh': land_thresh,
        'tracker': tracker,
        'contact':'Zhe Feng, zhe.feng@pnnl.gov',
        'created_on':time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    fillvalue = np.nan
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}
    dsout.to_netcdf(path=out_filename, mode='w', format='NETCDF4', encoding=encoding)
    print(f'Output saved: {out_filename}')

    # import pdb; pdb.set_trace()
