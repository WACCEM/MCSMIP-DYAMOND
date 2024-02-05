"""
Extract 2D variable center at each MCS track and save the output to a netCDF file
"""
import numpy as np
import xarray as xr
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import sys, os
import time
import yaml
import dask
from dask.distributed import Client, LocalCluster

def extract_env_2d(idatetime, mcs_lat, mcs_lon, ny, nx, varname):

    # Read environment file
    dse = xr.open_dataset(env_file).sel(time=idatetime)
    lat_e = dse['lat']
    lon_e = dse['lon']
    time_e = dse['time']

    # Predefine variable attributes dictionary
    VAR_attrs_dict = {
        'intqv': {'long_name': 'Total column water vapor', 'units': 'kg m**-2'},
    }
    # Get variable attribute from the dictionary
    VAR_attrs = VAR_attrs_dict[varname]

    # Number of tracks in the file
    ntracks_file = len(mcs_lat)
    # Make array to store output
    VAR_out = np.full((ntracks_file, 2*ny+1, 2*nx+1), np.NaN, dtype=np.float32)

    # Loop over each MCS
    for itrack in range(0, ntracks_file):
        # print(f'{itrack}: {mcs_lon[itrack]}')
        # MCS center lat/lon must not be a NaN
        if ~np.isnan(mcs_lat[itrack]) & ~np.isnan(mcs_lon[itrack]):        
            # Find closest grid point and time index
            lat_idx = np.abs(lat_e.data - mcs_lat[itrack]).argmin()
            lon_idx = np.abs(lon_e.data - mcs_lon[itrack]).argmin()
            # t_idx = np.abs(time_e.data - mcs_time[itrack]).argmin()
            
            # Select the region and time
            # Note index+1 is needed to include the last value in the range
            iVAR = dse[varname].isel(
                # time=t_idx, 
                lat=slice(lat_idx-ny, lat_idx+ny+1), 
                lon=slice(lon_idx-nx, lon_idx+nx+1),
            )
            # Check box size to avoid error (e.g. at date line)
            inx = iVAR.sizes['lon']
            iny = iVAR.sizes['lat']
            if (iny > 1) & (inx > 1):
                VAR_out[itrack,0:iny,0:inx] = iVAR.data

    # Put output variables to a dictionary for easier acceess
    out_dict = {'VAR':VAR_out, 'VAR_attrs':VAR_attrs}
    # import pdb; pdb.set_trace()
    print(f'Done processing: {idatetime}')

    return out_dict


if __name__ == "__main__":

    PHASE = sys.argv[1]
    runname = sys.argv[2]
    tracker = sys.argv[3]
    varname = sys.argv[4]

    # Config file contains input filenames for each model & environmental variable names
    config_file = f'/global/homes/f/feng045/program/mcsmip/dyamond/src/config_env_files_{PHASE}.yml'

    stats_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_stats/{PHASE}/{tracker}/'
    env_dir = f'/pscratch/sd/f/feng045/DYAMOND/{PHASE}/{runname}/envs/'
    out_dir = stats_dir
    out_filename = f'{out_dir}mcs_tracks_{PHASE}_{runname}_{varname}_2d.nc'

    mcs_file = f"{stats_dir}mcs_stats_{PHASE}_{runname}.nc"
    env_file = f"{env_dir}{PHASE}_{runname}_{varname}.nc"

    # Get inputs from configuration file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)
    if runname == 'OBS':
        nx = config[runname]['nx']
        ny = config[runname]['ny']
        dx = '0.25deg'
    else:
        nx = 30
        ny = 30
        dx = '0.1deg'
    # import pdb; pdb.set_trace()

    # Number of hours prior to initiation to save
    nhours = 24
    # Set max number of times to keep for each MCS track
    ntimes_max = 200
    # Dask workers and threads
    run_parallel = 1
    n_workers = 128
    threads_per_worker = 1
    dask_tmp_dir = '/tmp'

    # Read robust MCS statistics
    dsm = xr.open_dataset(mcs_file)
    # Subset MCS times to reduce array size
    # Most valid MCS data are within 0:ntimes_max
    dsm = dsm.isel(times=slice(0, ntimes_max))
    ntracks = dsm.sizes['tracks']
    # ntimes = dsm.sizes['times']
    rmcs_lat = dsm['meanlat']
    rmcs_lon = dsm['meanlon']

    # For OBS (ERA5), check if longitude is [-180~+180], if so convert it to [0~360]
    if (runname == 'OBS') & (np.nanmin(rmcs_lon) < 0):
        rmcs_lon = rmcs_lon % 360
        print('MCS longitudes are [-180~+180], converted to [0-360] to match ERA5.')

    # Get end times for all tracks
    rmcs_basetime = dsm['base_time']
    # Replace no track times with NaT 
    # this is due to no _FillValue attribute in base_time, causing the -9999 to be incorrectly decoded
    rmcs_basetime = rmcs_basetime.where(~np.isnan(rmcs_lat), np.datetime64('NaT'))
    # Sum over time dimension for valid basetime indices, -1 to get the last valid time index for each track
    # This is the end time index of each track (i.e. +1 equals the lifetime of each track)
    end_time_idx = np.sum(np.isfinite(rmcs_basetime), axis=1)-1
    # Apply fancy indexing to base_time: a tuple that indicates for each track, get the end time index
    end_basetime = rmcs_basetime[(np.arange(0,ntracks), end_time_idx)]
        
    # Get the min/max of all base_times
    min_basetime = np.nanmin(rmcs_basetime.isel(times=0).data)
    max_basetime = np.nanmax(end_basetime.data)
    # import pdb; pdb.set_trace()

    # Make a hourly DatetimeIndex that includes all tracks
    mcs_alldates = pd.date_range(start=min_basetime, end=max_basetime, freq='H')
    # import pdb; pdb.set_trace()

    # Select initiation time, and round to the nearest hour
    time0 = dsm['start_basetime'].dt.round('H')
    # time0 = rmcs_basetime.isel(times=0).dt.round('H')
    # Get initiation lat/lon    
    rmcs_lon0 = rmcs_lon.isel(times=0).data
    rmcs_lat0 = rmcs_lat.isel(times=0).data

    # Make an array to store the full time series
    nhours_full = nhours + ntimes_max - 1
    full_times = np.ndarray((ntracks, nhours_full), dtype='datetime64[ns]')
    full_lons = np.full((ntracks, nhours_full), np.NaN, dtype=np.float32)
    full_lats = np.full((ntracks, nhours_full), np.NaN, dtype=np.float32)

    # Get MCS track data numpy arrays for better performance
    rmcs_hour0 = time0.data
    rmcs_hours = rmcs_basetime.dt.round('H').data
    rmcs_lons = rmcs_lon.data
    rmcs_lats = rmcs_lat.data

    # Loop over each track
    for itrack in range(0, ntracks):
        # Calculate start/end times prior to initiation
        time0_start = rmcs_hour0[itrack] - pd.offsets.Hour(nhours-1)
        time0_end = rmcs_hour0[itrack] - pd.offsets.Hour(1)
        # Generate hourly time series leading up to -1 h before initiation
        prior_times = np.array(pd.date_range(time0_start, time0_end, freq='1H'))

        # Save full history of times
        full_times[itrack,0:nhours-1] = prior_times
        full_times[itrack,nhours-1:] = rmcs_hours[itrack,:]

        # Repeat initiation lat/lon by X hours (i.e., stay at the initiation location)
        ilon0 = np.repeat(rmcs_lon0[itrack], nhours-1)
        ilat0 = np.repeat(rmcs_lat0[itrack], nhours-1)
        # Save full history of lat/lon
        full_lons[itrack,0:nhours-1] = ilon0
        full_lons[itrack,nhours-1:] = rmcs_lons[itrack,:]
        full_lats[itrack,0:nhours-1] = ilat0
        full_lats[itrack,nhours-1:] = rmcs_lats[itrack,:]

    # Convert to Xarray DataArray
    coord_relativetimes = np.arange(-nhours+1, ntimes_max, 1)
    coord_relativetimes_attrs = {
        'description': 'Relative times for MCS lifecycle',
        'units': 'hour',       
    }
    ntimes_full = len(coord_relativetimes)
    coord_tracks = dsm['tracks']
    full_times = xr.DataArray(
        full_times, 
        coords={'tracks':coord_tracks, 'hours':coord_relativetimes}, 
        dims=('tracks','hours'),
    )

    # Total number of unique hours during the period
    nhours_unique = len(mcs_alldates)

    # TODO: change nhours_unique for short test
    # nhours_unique = 2
    print(f"Total number of hours: {nhours_unique}")

    # import pdb; pdb.set_trace()

    # Create a list to store matchindices for each ERA5 file
    trackindices_all = []
    timeindices_all = []
    results = []

    if run_parallel == 1:
        # Set Dask temporary directory for workers
        dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)


    # Loop over each unique MCS hour
    for ifile in range(nhours_unique):
        # We use mcs_alldates instead of full_times (extended pre-initiation) here
        # because we don't have data prior to the start date 
        idatetime = mcs_alldates[ifile]
        # print(filename)

        # Get all MCS tracks/times indices that are at the same time
        idx_track, idx_time = np.where(full_times == idatetime)

        # Save track/time indices for the current ERA5 file to the overall list
        trackindices_all.append(idx_track)
        timeindices_all.append(idx_time)

        # Get the track lat/lon/time values
        mcs_lat = full_lats[idx_track, idx_time]
        mcs_lon = full_lons[idx_track, idx_time]
        mcs_time = full_times.data[idx_track, idx_time]

        # Run in serial or parallel
        if run_parallel == 0:
            # Call function to calculate statistics
            result = extract_env_2d(idatetime, mcs_lat, mcs_lon, ny, nx, varname)
            results.append(result)
        elif run_parallel == 1:
            result = dask.delayed(extract_env_2d)(idatetime, mcs_lat, mcs_lon, ny, nx, varname)
            results.append(result)

    # Final returned outputs
    if run_parallel == 0:
        final_result = results
    elif run_parallel == 1:
        # Trigger dask computation
        final_result = dask.compute(*results)


    #-------------------------------------------------------------------------
    # Collect returned data and write to output
    #-------------------------------------------------------------------------
    print(f'Computation finished. Collecting returned outputs ...')

    # Create variables for saving output
    fillval = np.NaN
    VAR_out = np.full((ntracks, ntimes_full, 2*ny+1, 2*nx+1), fillval, dtype=np.float32)
    xcoords = np.arange(-nx, nx+1)
    ycoords = np.arange(-ny, ny+1)
    xcoords_attrs = {'long_name':'longitude grids center at MCS', 'units':dx}
    ycoords_attrs = {'long_name':'latitude grids center at MCS', 'units':dx}

    # Put the results to output track stats variables
    # Loop over each file (parallel return results)
    for ifile in range(nhours_unique):
    # for ifile in range(1):
        # Get the return results for this pixel file
        iVAR = final_result[ifile]
        if iVAR is not None:
            trackindices = trackindices_all[ifile]
            timeindices = timeindices_all[ifile]
            # Put variable to the MCS track array
            VAR_out[trackindices, timeindices, :, :] = iVAR['VAR']
    # Variable attributes
    VAR_attrs = final_result[0]['VAR_attrs']

    # Define output dataset
    var_dict = {
        f"{varname}": (['tracks', 'rel_times', 'y', 'x'], VAR_out, VAR_attrs),
    }
    # Define coordinate list
    coord_dict = {
        'tracks': (['tracks'], dsm['tracks'].data, dsm['tracks'].attrs),
        'rel_times': (['rel_times'], coord_relativetimes, coord_relativetimes_attrs),
        'y': (['y'], ycoords, ycoords_attrs),
        'x': (['x'], xcoords, xcoords_attrs),
    }
    # Define global attributes
    gattr_dict = {
        'Title': 'Extracted 2D environments for MCS',
        'lon_box_size': nx*2+1,
        'lat_box_size': ny*2+1,
        'Institution': 'Pacific Northwest National Laboratoy',
        'Contact': 'zhe.feng@pnnl.gov',
        'Created_on': time.ctime(time.time()),
    }

    # Define output Xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    print(f'Writing output file ...')
    dsout.to_netcdf(path=out_filename, mode='w', format='NETCDF4', 
                    unlimited_dims='tracks', encoding=encoding)
    print(f'Output saved as: {out_filename}')