"""
Match regridded GPM DPR rain rate data with IMERG and MCS masks, calculate rain rate histogram
anb save output to a netCDF file.
"""
import time
import numpy as np
import xarray as xr
import pandas as pd
import glob, os, sys

def find_closest_index(array, values):
    """
    Find closest indices from an array
    """
    idx = np.searchsorted(array, values)
    idx = np.clip(idx, 1, len(array) - 1)
    left = array[idx - 1]
    right = array[idx]
    idx -= values - left < right - values
    return idx

if __name__ == "__main__":

    PHASE = sys.argv[1]

    # Specify regions
    if PHASE == 'Summer':
        lon_bounds = [-180, 180]
        lat_bounds = [-15, 30]
        year = 2016
    if PHASE == 'Winter':
        lon_bounds = [-180, 180]
        lat_bounds = [-20, 15]
        year = 2020
    
    tracker = 'PyFLEXTRKR'

    # Ocean vs. Land threshold (%)
    ocean_thresh = 99
    land_thresh = 20

    dir_root = '/pscratch/sd/f/feng045/DYAMOND/'
    dir_dpr = f'{dir_root}GPM_DYAMOND/DPR/'
    dir_dpr_phase = f'{dir_dpr}{year}/'
    basename_dpr = '2A.GPM.DPR.V9-20211125.'
    out_filename = f'{dir_dpr}rainrate_hist_{PHASE}_DPR.nc'

    dir_imerg = '/pscratch/sd/f/feng045/DYAMOND/OLR_Precipitation_combined/'
    file_imerg = f'{dir_imerg}olr_pcp_{PHASE}_OBS.nc'
    file_imergv7 = f'{dir_imerg}olr_pcp_{PHASE}_OBSv7.nc'

    # Find all DPR files
    files_dpr = sorted(glob.glob(f'{dir_dpr_phase}{basename_dpr}*regridded.nc'))
    nfiles_dpr = len(files_dpr)
    print(f'Number of DPR files: {nfiles_dpr}')

    # MCS mask file
    mask_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask/{PHASE}/{tracker}/'
    mask_file6 = f'{mask_dir}mcs_mask_{PHASE}_OBS.nc'
    mask_file7 = f'{mask_dir}mcs_mask_{PHASE}_OBSv7.nc'

    # Landmask file
    file_lm = f'{dir_root}maps/IMERG_landmask_180W-180E_60S-60N.nc'

    # Read landmask
    ds_lm = xr.open_dataset(file_lm).sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    )
    
    # Read MCS mask data (IMERG v6)
    dsm6 = xr.open_dataset(mask_file6).sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    )

    # Read MCS mask data (IMERG v7)
    dsm7 = xr.open_dataset(mask_file7).sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    )

    # Read IMERG v6 data
    ds_i6 = xr.open_dataset(file_imerg).sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    )
    
    # Read IMERG v7 data
    ds_i7 = xr.open_dataset(file_imergv7).sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    )
    
    # Read in all DPR files, concatenate by time
    ds_dpr = xr.open_mfdataset(files_dpr, combine='nested', concat_dim='time').sel(
        lon=slice(lon_bounds[0], lon_bounds[1]), 
        lat=slice(lat_bounds[0], lat_bounds[1]),
    )
    
    # Find the closest indices in IMERG v6 for each time in DPR
    closest_indices = xr.apply_ufunc(
        find_closest_index,
        ds_i6['time'].values,
        ds_dpr['time'].values,
        vectorize=True,
        dask='parallelized',
        output_dtypes=[int]
    )
    # Extract the closest times from IMERG
    closest_times6 = ds_i6['time'].isel(time=closest_indices)

    # Find the closest indices in IMERG v7 for each time in DPR
    closest_indices7 = xr.apply_ufunc(
        find_closest_index,
        ds_i7['time'].values,
        ds_dpr['time'].values,
        vectorize=True,
        dask='parallelized',
        output_dtypes=[int]
    )
    # Extract the closest times from IMERG
    closest_times7 = ds_i7['time'].isel(time=closest_indices7)

    # Find the closest indices in MCS mask v6 for each time in DPR
    closest_indices = xr.apply_ufunc(
        find_closest_index,
        dsm6['time'].values,
        ds_dpr['time'].values,
        vectorize=True,
        dask='parallelized',
        output_dtypes=[int]
    )
    # Extract the closest times from IMERG
    closest_times6_mcs = ds_i6['time'].isel(time=closest_indices)

    # Find the closest indices in MCS mask v7 for each time in DPR
    closest_indices = xr.apply_ufunc(
        find_closest_index,
        dsm7['time'].values,
        ds_dpr['time'].values,
        vectorize=True,
        dask='parallelized',
        output_dtypes=[int]
    )
    # Extract the closest times from IMERG
    closest_times7_mcs = ds_i7['time'].isel(time=closest_indices)

    # Subset MCS mask v6 times to the closest DPR times
    dsm6_sub = dsm6.sel(time=closest_times6_mcs)
    # Subset MCS mask v7 times to the closest DPR times
    dsm7_sub = dsm6.sel(time=closest_times7_mcs)
    # Subset IMERG v6 times to the closest DPR times
    ds_i6_sub = ds_i6.sel(time=closest_times6)
    # Subset IMERG v7 times to the closest DPR times
    ds_i7_sub = ds_i7.sel(time=closest_times7)
    
    # Replace DPR time with the closest times from IMERG as the coordinate
    ds_dpr_reindexed = ds_dpr.assign_coords(time=closest_times6)
    
    # Combine the DataSets, aligning them on the new time coordinates
    ds6 = xr.merge([ds_i6_sub, dsm6_sub, ds_dpr_reindexed])
    # Add landseamask to the DataSet
    ds6['landseamask'] = ds_lm.landseamask
    
    # Combine the DataSets, aligning them on the new time coordinates
    ds7 = xr.merge([ds_i7_sub, dsm7_sub, ds_dpr_reindexed])
    # Add landseamask to the DataSet
    ds7['landseamask'] = ds_lm.landseamask
    print(f'Finished matching & combining datasets.')
    
    ####################################################################
    # Masking DataArrays for DPR swaths, land/ocean, MCS
    ####################################################################
    print(f'Masking data ...')
    #-------------------------------------------------------------------
    # DPR
    #-------------------------------------------------------------------
    # Separate DPR ocean vs. land precipitation
    pcp_dpr_o = ds6.precipitation_dpr.where(ds6.landseamask >= ocean_thresh)
    pcp_dpr_l = ds6.precipitation_dpr.where(ds6.landseamask <= land_thresh)

    # Separate DPR ocean vs. land MCS precipitation
    mcs_mask6 = ds6.mcs_mask > 0
    mcspcp_dpr = ds6.precipitation_dpr.where(mcs_mask6)
    mcspcp_dpr_o = ds6.precipitation_dpr.where((ds6.landseamask >= ocean_thresh) & (mcs_mask6))
    mcspcp_dpr_l = ds6.precipitation_dpr.where((ds6.landseamask <= land_thresh) & (mcs_mask6))

    #-------------------------------------------------------------------
    # IMERG v6
    #-------------------------------------------------------------------
    # Separate IMERG data to within and outside of DPR swaths
    pcp_imerg6_in = ds6.precipitation.where(ds6.precipitation_dpr >= 0)
    pcp_imerg6_out = ds6.precipitation.where(np.isnan(ds6.precipitation_dpr))

    # Separate ocean vs. land precipitation
    # Ocean
    pcp_imerg6_o_in = ds6.precipitation.where((ds6.precipitation_dpr >= 0) & (ds6.landseamask >= ocean_thresh))
    pcp_imerg6_o_out = ds6.precipitation.where((np.isnan(ds6.precipitation_dpr)) & (ds6.landseamask >= ocean_thresh))

    # Land
    pcp_imerg6_l_in = ds6.precipitation.where((ds6.precipitation_dpr >= 0) & (ds6.landseamask <= land_thresh))
    pcp_imerg6_l_out = ds6.precipitation.where((np.isnan(ds6.precipitation_dpr)) & (ds6.landseamask <= land_thresh))

    # Separate ocean vs. land MCS precipitation
    mcspcp_imerg6 = ds6.precipitation.where(mcs_mask6)
    mcspcp_imerg6_o = ds6.precipitation.where((ds6.landseamask >= ocean_thresh) & (mcs_mask6))
    mcspcp_imerg6_l = ds6.precipitation.where((ds6.landseamask <= land_thresh) & (mcs_mask6))

    #-------------------------------------------------------------------
    # IMERG v7
    #-------------------------------------------------------------------
    # Separate IMERG data to within and outside of DPR swaths
    pcp_imerg7_in = ds7.precipitation.where(ds7.precipitation_dpr >= 0)
    pcp_imerg7_out = ds7.precipitation.where(np.isnan(ds7.precipitation_dpr))

    # Separate ocean vs. land precipitation
    # Ocean
    pcp_imerg7_o_in = ds7.precipitation.where((ds7.precipitation_dpr >= 0) & (ds7.landseamask >= ocean_thresh))
    pcp_imerg7_o_out = ds7.precipitation.where((np.isnan(ds7.precipitation_dpr)) & (ds7.landseamask >= ocean_thresh))

    # Land
    pcp_imerg7_l_in = ds7.precipitation.where((ds7.precipitation_dpr >= 0) & (ds7.landseamask <= land_thresh))
    pcp_imerg7_l_out = ds7.precipitation.where((np.isnan(ds7.precipitation_dpr)) & (ds7.landseamask <= land_thresh))

    # Separate ocean vs. land MCS precipitation
    mcs_mask7 = ds7.mcs_mask > 0
    # mcspcp_imerg7 = ds7.precipitation.where(mcs_mask7)
    mcspcp_imerg7_o = ds7.precipitation.where((ds7.landseamask >= ocean_thresh) & (mcs_mask7))
    mcspcp_imerg7_l = ds7.precipitation.where((ds7.landseamask <= land_thresh) & (mcs_mask7))


    ####################################################################
    # Calculate histogram
    ####################################################################
    # Set up rain rate bins
    bins_pcp = np.arange(1, 301, 1)
    # bins_pcp = np.arange(1, 301, 2)
    pcp_range = (np.min(bins_pcp), np.max(bins_pcp))

    # Rain rate bin center values
    bins_pcp_c = bins_pcp[:-1]

    #-------------------------------------------------------------------
    # DPR
    #-------------------------------------------------------------------
    print(f'Computing DPR histogram ...')
    pcp_dpr_hist, bins = np.histogram(ds6.precipitation_dpr, bins=bins_pcp, range=pcp_range, density=False)

    pcp_dpr_o_hist, bins = np.histogram(pcp_dpr_o, bins=bins_pcp, range=pcp_range, density=False)
    pcp_dpr_l_hist, bins = np.histogram(pcp_dpr_l, bins=bins_pcp, range=pcp_range, density=False)

    mcspcp_dpr_hist, bins = np.histogram(mcspcp_dpr, bins=bins_pcp, range=pcp_range, density=False)
    mcspcp_dpr_o_hist, bins = np.histogram(mcspcp_dpr_o, bins=bins_pcp, range=pcp_range, density=False)
    mcspcp_dpr_l_hist, bins = np.histogram(mcspcp_dpr_l, bins=bins_pcp, range=pcp_range, density=False)

    #-------------------------------------------------------------------
    # IMERG v6
    #-------------------------------------------------------------------
    print(f'Computing IMERG v6 histogram ...')
    pcp_imerg6_in_hist, bins = np.histogram(pcp_imerg6_in, bins=bins_pcp, range=pcp_range, density=False)
    pcp_imerg6_out_hist, bins = np.histogram(pcp_imerg6_out, bins=bins_pcp, range=pcp_range, density=False)

    pcp_imerg6_o_in_hist, bins = np.histogram(pcp_imerg6_o_in, bins=bins_pcp, range=pcp_range, density=False)
    pcp_imerg6_o_out_hist, bins = np.histogram(pcp_imerg6_o_out, bins=bins_pcp, range=pcp_range, density=False)

    pcp_imerg6_l_in_hist, bins = np.histogram(pcp_imerg6_l_in, bins=bins_pcp, range=pcp_range, density=False)
    pcp_imerg6_l_out_hist, bins = np.histogram(pcp_imerg6_l_out, bins=bins_pcp, range=pcp_range, density=False)

    mcspcp_imerg6_o_hist, bins = np.histogram(mcspcp_imerg6_o, bins=bins_pcp, range=pcp_range, density=False)
    mcspcp_imerg6_l_hist, bins = np.histogram(mcspcp_imerg6_l, bins=bins_pcp, range=pcp_range, density=False)

    #-------------------------------------------------------------------
    # IMERG v7
    #-------------------------------------------------------------------
    print(f'Computing IMERG v7 histogram ...')
    pcp_imerg7_in_hist, bins = np.histogram(pcp_imerg7_in, bins=bins_pcp, range=pcp_range, density=False)
    pcp_imerg7_out_hist, bins = np.histogram(pcp_imerg7_out, bins=bins_pcp, range=pcp_range, density=False)

    pcp_imerg7_o_in_hist, bins = np.histogram(pcp_imerg7_o_in, bins=bins_pcp, range=pcp_range, density=False)
    pcp_imerg7_o_out_hist, bins = np.histogram(pcp_imerg7_o_out, bins=bins_pcp, range=pcp_range, density=False)

    pcp_imerg7_l_in_hist, bins = np.histogram(pcp_imerg7_l_in, bins=bins_pcp, range=pcp_range, density=False)
    pcp_imerg7_l_out_hist, bins = np.histogram(pcp_imerg7_l_out, bins=bins_pcp, range=pcp_range, density=False)

    mcspcp_imerg7_o_hist, bins = np.histogram(mcspcp_imerg7_o, bins=bins_pcp, range=pcp_range, density=False)
    mcspcp_imerg7_l_hist, bins = np.histogram(mcspcp_imerg7_l, bins=bins_pcp, range=pcp_range, density=False)


    #-------------------------------------------------------------------------
    # Write output file
    #-------------------------------------------------------------------------
    # Define xarray output dataset
    print('Writing output to netCDF file ...')
    bins_pcp_attrs = {'units': 'mm h-1'}
    tb_attrs = {'long_name': 'Tb histogram', 'units':'count'}
    totpcp_dpr_attrs = {'long_name': 'DPR total precipitation histogram', 'units':'count'}
    totpcp_dpr_o_attrs = {'long_name': 'DPR ocean precipitation histogram', 'units':'count'}
    totpcp_dpr_l_attrs = {'long_name': 'DPR land precipitation histogram', 'units':'count'}
    totpcp_imerg6_attrs = {'long_name': 'IMERGv6 total precipitation histogram', 'units':'count'}
    totpcp_imerg6_o_attrs = {'long_name': 'IMERGv6 ocean precipitation histogram', 'units':'count'}
    totpcp_imerg6_l_attrs = {'long_name': 'IMERGv6 land precipitation histogram', 'units':'count'}
    totpcp_imerg7_attrs = {'long_name': 'IMERGv7 total precipitation histogram', 'units':'count'}
    totpcp_imerg7_o_attrs = {'long_name': 'IMERGv7 ocean precipitation histogram', 'units':'count'}
    totpcp_imerg7_l_attrs = {'long_name': 'IMERGv7 land precipitation histogram', 'units':'count'}

    mcspcp_dpr_attrs = {'long_name': 'DPR MCS precipitation histogram', 'units':'count'}
    mcspcp_dpr_o_attrs = {'long_name': 'DPR ocean MCS precipitation histogram', 'units':'count'}
    mcspcp_dpr_l_attrs = {'long_name': 'DPR land MCS precipitation histogram', 'units':'count'}

    mcspcp_imerg6_o_attrs = {'long_name': 'IMERGv6 ocean MCS precipitation histogram', 'units':'count'}
    mcspcp_imerg6_l_attrs = {'long_name': 'IMERGv6 land MCS precipitation histogram', 'units':'count'}

    mcspcp_imerg7_o_attrs = {'long_name': 'IMERGv7 ocean MCS precipitation histogram', 'units':'count'}
    mcspcp_imerg7_l_attrs = {'long_name': 'IMERGv7 land MCS precipitation histogram', 'units':'count'}

    var_dict = {
        'total_pcp_dpr': (['bins_pcp'], pcp_dpr_hist, totpcp_dpr_attrs),
        'total_pcp_imerg_v6_in': (['bins_pcp'], pcp_imerg6_in_hist, totpcp_imerg6_attrs),
        'total_pcp_imerg_v7_in': (['bins_pcp'], pcp_imerg7_in_hist, totpcp_imerg7_attrs),
        'total_pcp_imerg_v6_out': (['bins_pcp'], pcp_imerg6_out_hist, totpcp_imerg6_attrs),
        'total_pcp_imerg_v7_out': (['bins_pcp'], pcp_imerg7_out_hist, totpcp_imerg7_attrs),

        'total_pcp_o_dpr': (['bins_pcp'], pcp_dpr_o_hist, totpcp_dpr_attrs),
        'total_pcp_o_imerg_v6_in': (['bins_pcp'], pcp_imerg6_o_in_hist, totpcp_imerg6_o_attrs),
        'total_pcp_o_imerg_v7_in': (['bins_pcp'], pcp_imerg7_o_in_hist, totpcp_imerg7_o_attrs),
        'total_pcp_o_imerg_v6_out': (['bins_pcp'], pcp_imerg6_o_out_hist, totpcp_imerg6_o_attrs),
        'total_pcp_o_imerg_v7_out': (['bins_pcp'], pcp_imerg7_o_out_hist, totpcp_imerg7_o_attrs),
        
        'total_pcp_l_dpr': (['bins_pcp'], pcp_dpr_l_hist, totpcp_dpr_attrs),
        'total_pcp_l_imerg_v6_in': (['bins_pcp'], pcp_imerg6_l_in_hist, totpcp_imerg6_l_attrs),
        'total_pcp_l_imerg_v7_in': (['bins_pcp'], pcp_imerg7_l_in_hist, totpcp_imerg7_l_attrs),
        'total_pcp_l_imerg_v6_out': (['bins_pcp'], pcp_imerg6_l_out_hist, totpcp_imerg6_l_attrs),
        'total_pcp_l_imerg_v7_out': (['bins_pcp'], pcp_imerg7_l_out_hist, totpcp_imerg7_l_attrs),
        
        'mcs_pcp_dpr': (['bins_pcp'], mcspcp_dpr_hist, mcspcp_dpr_attrs),
        'mcs_pcp_o_dpr': (['bins_pcp'], mcspcp_dpr_o_hist, mcspcp_dpr_o_attrs),
        'mcs_pcp_o_imerg_v6': (['bins_pcp'], mcspcp_imerg6_o_hist, mcspcp_dpr_o_attrs),
        'mcs_pcp_o_imerg_v7': (['bins_pcp'], mcspcp_imerg7_o_hist, mcspcp_dpr_o_attrs),
        'mcs_pcp_l_dpr': (['bins_pcp'], mcspcp_dpr_l_hist, mcspcp_dpr_l_attrs),
        'mcs_pcp_l_imerg_v6': (['bins_pcp'], mcspcp_imerg6_l_hist, mcspcp_dpr_l_attrs),
        'mcs_pcp_l_imerg_v7': (['bins_pcp'], mcspcp_imerg7_l_hist, mcspcp_dpr_l_attrs),
        
        # 'ocean_pcp_dpr': (['bins_pcp'], pcp_dpr_o_hist, totpcp_dpr_attrs),
        # 'ocean_pcp_imerg_v6_in': (['bins_pcp'], pcp_imerg6_o_in_hist, totpcp_imerg6_o_attrs),
        # 'ocean_pcp_imerg_v7_in': (['bins_pcp'], pcp_imerg7_o_in_hist, totpcp_imerg7_o_attrs),
        # 'ocean_pcp_imerg_v6_out': (['bins_pcp'], pcp_imerg6_o_out_hist, totpcp_imerg6_o_attrs),
        # 'ocean_pcp_imerg_v7_out': (['bins_pcp'], pcp_imerg7_o_out_hist, totpcp_imerg7_o_attrs),
        
        # 'land_pcp_dpr': (['bins_pcp'], pcp_dpr_l_hist, totpcp_dpr_attrs),
        # 'land_pcp_imerg_v6_in': (['bins_pcp'], pcp_imerg6_l_in_hist, totpcp_imerg6_l_attrs),
        # 'land_pcp_imerg_v7_in': (['bins_pcp'], pcp_imerg7_l_in_hist, totpcp_imerg7_l_attrs),
        # 'land_pcp_imerg_v6_out': (['bins_pcp'], pcp_imerg6_l_out_hist, totpcp_imerg6_l_attrs),
        # 'land_pcp_imerg_v7_out': (['bins_pcp'], pcp_imerg7_l_out_hist, totpcp_imerg7_l_attrs),
        
        # 'ocean_mcs_pcp_dpr': (['bins_pcp'], mcspcp_dpr_o_hist, mcspcp_dpr_o_attrs),   
        # 'ocean_mcs_pcp_imerg_v6': (['bins_pcp'], mcspcp_imerg6_o_hist, mcspcp_dpr_o_attrs),
        # 'ocean_mcs_pcp_imerg_v7': (['bins_pcp'], mcspcp_imerg7_o_hist, mcspcp_dpr_o_attrs),
        # 'land_mcs_pcp_dpr': (['bins_pcp'], mcspcp_dpr_l_hist, mcspcp_dpr_l_attrs),
        # 'land_mcs_pcp_imerg_v6': (['bins_pcp'], mcspcp_imerg6_l_hist, mcspcp_dpr_l_attrs),
        # 'land_mcs_pcp_imerg_v7': (['bins_pcp'], mcspcp_imerg7_l_hist, mcspcp_dpr_l_attrs),
        
    }
    coord_dict = {
        'bins_pcp': (['bins_pcp'], bins_pcp_c, bins_pcp_attrs),
    }
    gattr_dict = {
        'title': 'GPM DPR precipitation histogram',
        'lon_bounds': lon_bounds,
        'lat_bounds': lat_bounds,
        # 'tracker': tracker,
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