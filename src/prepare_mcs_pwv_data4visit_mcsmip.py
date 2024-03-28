"""
Prepare MCSMIP MCS & PWV data for visulization in VisIt.

Example data for VisIt:
//global/cfs/cdirs/m1867/zfeng/MCSMIP/DYAMOND/Summer/data4visit

Author: Zhe Feng, zhe.feng@pnnl.gov
History:
03/26/2024 - Written.
"""

import numpy as np
import glob, os, sys
import xarray as xr
import pandas as pd

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


if __name__=='__main__':

    # Get inputs
    PHASE = sys.argv[1]
    tracker = sys.argv[2]
    run_name = sys.argv[3]

    if PHASE == 'Summer':
        start_datetime = '2016-08-27T00'
        end_datetime = '2016-08-29T00'
    elif PHASE == 'Winter':
        start_datetime = '2020-02-05T00'
        end_datetime = '2020-02-07T00'

    # Data directories
    rootdir = '/pscratch/sd/f/feng045/DYAMOND/'
    mcs_dir = f'{rootdir}mcs_mask/{PHASE}/{tracker}/'
    tbpcp_dir = f'{rootdir}OLR_Precipitation_combined/'
    pwv_dir = f'{rootdir}/{PHASE}/{run_name}/envs/'
    outdir = f'{rootdir}/{PHASE}/data4visit/'
    # Data filenames
    mask_file = f'{mcs_dir}mcs_mask_{PHASE}_{run_name}.nc'
    tbpcp_file = f'{tbpcp_dir}olr_pcp_{PHASE}_{run_name}.nc'
    pwv_file = f'{pwv_dir}{PHASE}_{run_name}_intqv.nc'
    out_mcsbasename = f'mcs4visit_{PHASE}_{run_name}_'
    out_pwvbasename = f'pwv4visit_{PHASE}_{run_name}_'

    os.makedirs(outdir, exist_ok=True)

    drop_varlist = ['longitude', 'latitude']

    # Create a range of dates
    dates = pd.date_range(start=start_datetime, end=end_datetime, freq='D')
    mcs_dates = dates.strftime('%Y%m%d')
    outdate1 = dates[0].strftime('%Y%m%d')
    outdate2 = dates[-1].strftime('%Y%m%d')
    out_mcsfilename = f'{outdir}{out_mcsbasename}{outdate1}_{outdate2}.nc'
    out_pwvfilename = f'{outdir}{out_pwvbasename}{outdate1}_{outdate2}.nc'

    # Check required input files
    if os.path.isfile(mask_file) == False:
        print(f'ERROR: mask file does not exist: {mask_file}')
        sys.exit(f'Code will exist now.')
    if os.path.isfile(tbpcp_file) == False:
        print(f'ERROR: tbpcp file does not exist: {tbpcp_file}')
        sys.exit(f'Code will exist now.')
    
    # Check PWV file
    if os.path.isfile(pwv_file) == True:
        # Read PWV file and subset times
        dspwv = xr.open_dataset(pwv_file).sel(time=slice(start_datetime, end_datetime))
        # Set encoding/compression for all variables
        comp = dict(zlib=True, dtype='float32')
        encoding = {var: comp for var in dspwv.data_vars}
        # Update time variable dtype as 'double' for better precision
        bt_dict = {'time': {'zlib':True, 'dtype':'float64'}}
        encoding.update(bt_dict)
        # Write to netcdf file
        dspwv.to_netcdf(path=out_pwvfilename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
        print(f'PWV file saved: {out_pwvfilename}')
    else:
        print(f'PWV file does not exist: {pwv_file}')


    # Read precipitation file and subset times
    print(f'Reading tbpcp file: {tbpcp_file}')
    dsr = xr.open_dataset(tbpcp_file).sel(time=slice(start_datetime, end_datetime))
    # Convert CFTimeIndex to Pandas DatetimeInex
    # This gets around issues with time coordinates in cftime.DatetimeNoLeap format (e.g., SCREAM)
    if run_name == 'SCREAM':
        dsr_datetimeindex = dsr.indexes['time'].to_datetimeindex()
        # Replace the original time coordinate
        dsr = dsr.assign_coords({'time': dsr_datetimeindex})
    ntimes_pcp = dsr.sizes['time']
    out_time = dsr['time'].isel(time=0)
    lon = dsr['lon']
    lat = dsr['lat']

    # Read MCS mask file
    print(f'Reading MCS mask file: {mask_file}')
    dsm = xr.open_dataset(mask_file, drop_variables=drop_varlist, mask_and_scale=False)
    # Convert CFTimeIndex to Pandas DatetimeInex
    if dsm['time'].encoding.get('calendar') == 'noleap':
        dsm_datetimeindex = dsm.indexes['time'].to_datetimeindex()
        # Replace the original time coordinate
        dsm = dsm.assign_coords({'time': dsm_datetimeindex})
    
    # Subset mask file for the times
    dsm = dsm.sel(time=dsr['time'], method='nearest')
    ntimes_mask = dsm.sizes['time']
    if (ntimes_mask != ntimes_pcp):
        print(f'ERROR: Subset times in mcs_mask file is NOT the same with precipitation file.')
        sys.exit(f'Code will exit now.')

    # Combine DataSets, subset region
    ds = xr.merge([dsr, dsm], compat='no_conflicts')
    print(f'Finished combining two datasets.')

    # Red Tb for OBS data, convert OLR to Tb for model data
    if 'OBS' in run_name:
        tb = ds['Tb']
        # Replace NaN with a large number
        ds['Tb'] = tb.where(~np.isnan(tb), other=400)
    else:
        olr = ds['olr']
        tb = olr_to_tb(olr)
        tb.attrs = {'long_name': 'Brightness temperature', 'units': 'K'}
        # Add to DataSet
        ds['Tb'] = tb
        # Drop OLR variable
        ds = ds.drop_vars(['olr'])

    # Filter non-MCS tb, replace NaN with a large number
    tb_mcs = tb.where(ds['mcs_mask'] > 0, other=400)
    # Filter non-CCS tb, replace NaN with a large number
    tb_ccs = tb.where(ds['cloud_mask'] > 0, other=400)
    # Add to DataSet
    ds['Tb_mcs'] = tb_mcs
    # ds['Tb_ccs'] = tb_ccs
    
    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in ds.data_vars}
    # Update time variable dtype as 'double' for better precision
    bt_dict = {'time': {'zlib':True, 'dtype':'float64'}}
    encoding.update(bt_dict)
    # Write to netcdf file
    ds.to_netcdf(path=out_mcsfilename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'MCS file saved: {out_mcsfilename}')

    # import pdb; pdb.set_trace()