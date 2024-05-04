"""
Counts unique tracks on a grid map and save output to a netCDF file.
"""
import numpy as np
import pandas as pd
import xarray as xr
from itertools import chain
import sys, time, os

#--------------------------------------------------------------------------
def get_unique_placements(track_num, lat, lon, ntimes):
    """
    Get unique pairs of lat/lon for a track
    """
    # Put all lat/lon pairs over each time for a track into an array
    this_row = np.array([[lat[track_num, tt], lon[track_num, tt]] for tt in range(0, ntimes)])
    # Return the unique pairs (axis=0)
    return np.array(np.unique(this_row, axis=0))

#--------------------------------------------------------------------------
def count_unique_tracks(lat, lon, xbins, ybins):
    
    ntracks, ntimes = lat.shape

    # A function to loop over tracks
    get_unique = lambda D: get_unique_placements(D, lat, lon, ntimes)

    # Loop over each track and get the unique pairs of lat/lon
    all_uniques = list(map(get_unique, np.arange(0, ntracks)))

    # Flatten the list of lat/lon pairs (using chain), and convert into an array
    unique_latlon = np.array(list(chain(*all_uniques)))

    # Count number on map using histogram2d
    ranges = [[min(ybins), max(ybins)], [min(xbins), max(xbins)]]
    hist2d, yedges, xedges = np.histogram2d(unique_latlon[:,0], unique_latlon[:,1], bins=[ybins, xbins], range=ranges)

    return hist2d

#--------------------------------------------------------------------------
def count_tracks_on_gridmap(
        PHASE,
        runname,
        tracker,
        start_datetime,
        end_datetime,
):

    # # Get inputs
    # PHASE = sys.argv[1]
    # runname = sys.argv[2]
    # tracker = sys.argv[3]
    # start_datetime = sys.argv[4]
    # end_datetime = sys.argv[5]

    # Track lat/lon variable names
    lat_varname = 'meanlat'
    lon_varname = 'meanlon'
    # lat_varname = 'lat_mintb'
    # lon_varname = 'lon_mintb'

    # Mask filename
    in_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_stats/{PHASE}/{tracker}/'
    in_file = f'{in_dir}mcs_stats_{PHASE}_{runname}.nc'
    out_dir = in_dir
    out_filename = f'{out_dir}mcs_counts_gridmap_{PHASE}_{runname}.nc'

    # Check input file
    if os.path.isfile(in_file):

        # Convert dates to Pandas datetime
        start_date = pd.to_datetime(start_datetime)
        end_date = pd.to_datetime(end_datetime)

        # Read input data
        ds = xr.open_dataset(in_file)
        # Get track initial time valuees
        base_time = ds.base_time.load()
        starttime = base_time.isel(times=0)
        # Count tracks within the specified period
        ntracks = len(starttime.where((starttime >= start_date) & (starttime <= end_date), drop=True))
        print(f'Number of tracks ({runname}): {ntracks}')

        # Round the lat/lon to the nearest integer, 
        # This way the lat/lon are calculated at the precision of 1 degree (i.e., count on a 1x1 degree grid)
        # Note: counting using histogram2d only works on 1x1 degree grid (unique lat/lon are round to integer)
        rlat = ds[lat_varname].where((starttime >= start_date) & \
            (starttime <= end_date), drop=False).load().round().data
        rlon = ds[lon_varname].where((starttime >= start_date) & \
            (starttime <= end_date), drop=False).load().round().data

        # rlat0 = ds[lat_varname].isel(times=0).where((starttime >= start_date) & \
        #     (starttime <= end_date), drop=False).load().round().data
        # rlon0 = ds[lon_varname].isel(times=0).where((starttime >= start_date) & \
        #     (starttime <= end_date), drop=False).load().round().data

        # # Specify grid
        # if PHASE == 'Winter':
        #     ranges = [[-60,60], [0,360]]
        #     xbins = np.arange(0, 360.1, 1)
        #     ybins = np.arange(-60., 60.1, 1)
        # # ranges = [[-60,60], [0,360]]
        # # xbins = np.arange(0, 360.1, 1)
        # elif PHASE == 'Summer':
        #     ranges = [[-60,60], [-180,180]]
        #     xbins = np.arange(-180, 180.1, 1)
        #     ybins = np.arange(-60., 60.1, 1)
        # else:
        #     print(f'Error, unknown PHASE: {PHASE}')
        #     sys.exit()
            
        # Specify grid
        xbins = np.arange(-180, 180.1, 1)
        ybins = np.arange(-60., 60.1, 1)

        # Total tracks count
        track_counts = count_unique_tracks(rlat, rlon, xbins, ybins)

        # Calculate lat/lon bin center value
        xbins_c = xbins[:-1] + np.diff(xbins)/2.
        ybins_c = ybins[:-1] + np.diff(ybins)/2.

        # Write output file
        var_dict = {
            'track_counts': (['time', 'lat', 'lon'], np.expand_dims(track_counts, axis=0)),
        }
        coord_dict = {
            'time': (['time'], np.expand_dims(start_date, axis=0)),
            'lat': (['lat'], ybins_c),
            'lon': (['lon'], xbins_c),
            'lat_bnds': (['lat_bnds'], ybins),
            'lon_bnds': (['lon_bnds'], xbins),
        }
        gattr_dict = {
            'Title': 'MCS track counts on grid',
            'phase': f'{PHASE}',
            'tracker': f'{tracker}',
            'start_date': start_datetime,
            'end_date': end_datetime,
            'contact':'Zhe Feng, zhe.feng@pnnl.gov',
            'created_on':time.ctime(time.time()),
        }
        dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
        dsout['lon'].attrs['long_name'] = 'Longitude grid center value'
        dsout['lon'].attrs['units'] = 'degree'
        dsout['lat'].attrs['long_name'] = 'Latitude grid center value'
        dsout['lat'].attrs['units'] = 'degree'
        dsout['lon_bnds'].attrs['long_name'] = 'Longitude grid bounds'
        dsout['lon_bnds'].attrs['units'] = 'degree'
        dsout['lat_bnds'].attrs['long_name'] = 'Latitude grid grid bounds'
        dsout['lat_bnds'].attrs['units'] = 'degree'
        dsout['track_counts'].attrs['long_name'] = 'Track counts on gridded map'
        dsout['track_counts'].attrs['units'] = 'count'

        fillvalue = np.nan
        # Set encoding/compression for all variables
        comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
        encoding = {var: comp for var in dsout.data_vars}
        # Write output
        dsout.to_netcdf(path=out_filename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
        print(f'Output saved: {out_filename}')
    
    else:
        print(f'{in_file} does NOT exist.')

    return out_filename
    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    
    PHASE = sys.argv[1]
    tracker = sys.argv[2]

    # DYAMOND phase start date
    if PHASE == 'Summer':
        # start_date = '2016-08-01T00'
        start_datetime = '2016-08-10T00'
        end_datetime = '2016-09-10T00'
    elif PHASE == 'Winter':
        # start_date = '2020-01-20T00'
        start_datetime = '2020-02-01T00'
        end_datetime = '2020-03-01T00'

    # Model names
    if (PHASE == 'Summer'):
        # runnames = ['MPAS']
        runnames = [
            'ARPEGE',
            'FV3',
            'IFS',
            'MPAS',
            'NICAM',
            'OBS',
            'OBSv7',
            'SAM',
            'SCREAMv1',
            'UM',
        ]
    elif (PHASE == 'Winter'):
        # runnames=['SCREAM']
        runnames = [
            'ARPEGE',
            'GEOS',
            'GRIST',
            'ICON',
            'IFS',
            'MPAS',
            # 'NICAM',
            'OBS',
            'OBSv7',
            'SAM',
            'SCREAM',
            'SCREAMv1',
            'UM',
            'XSHiELD',
        ]
    else:
        print(f'Unknown PHASE: {PHASE}')

    # Loop over list
    for run in runnames:
        outfile = count_tracks_on_gridmap(PHASE, run, tracker, start_datetime, end_datetime)
        # import pdb; pdb.set_trace()


