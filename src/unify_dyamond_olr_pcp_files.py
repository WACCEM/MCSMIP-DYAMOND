"""
Unify the DYAMOND OLR/precipitation files:
- Rename the coordinate names to: time, lon, lat
- Replace the lat, lon coordinate values with a reference grid file
- Rename the variable names to: olr, precipitation
"""
# import numpy as np
import glob, sys, os
import time
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster, wait
from pyflextrkr.ft_utilities import load_config

def harmonize_file(filename):
    # Get datetime from filename
    fn = os.path.basename(filename)
    # Model filename: basename_yyyymoddhh.nc
    # OBS filename: merg_2016080100_4km-pixel.nc
    if runname != 'OBS':
        fn_datetime = fn[len(databasename):-3]
    else:
        fn_datetime = fn[len(databasename):-13]

    # Get variable names from config
    time_coordname = config['time_coordname']
    x_coordname = config['x_coordname']
    y_coordname = config['y_coordname']
    olr_varname = config.get('olr_varname', None)
    pcp_varname = config['pcp_varname']
    if olr_varname is None:
        olr_varname = config.get('tb_varname')

    # Read lon from reference grid
    dsref = xr.open_dataset(ref_grid)
    lon_ref = dsref['lon']
    lat_ref = dsref['lat']

    # Read input data
    drop_vars = ['gw', 'lat_bnds', 'lon_bnds', 'area']
    ds = xr.open_dataset(filename, drop_variables=drop_vars)
    # Change time coordinate encoding
    ds[time_coordname].encoding['units'] = f'hours since {start_datetime}'
    # Round down the time to the nearest hour
    t_coord_new = ds[time_coordname].dt.floor('H')
    # Replace the lat/lon coordinates
    ds = ds.assign_coords({time_coordname:t_coord_new, x_coordname:lon_ref, y_coordname:lat_ref})
    # Rename dimensions
    # This suppresses "UserWarning: rename 'lon' to 'lon' does not create an index anymore."
    # ds = ds.swap_dims({
    #     x_coordname: 'lon',
    #     y_coordname: 'lat',
    # })
    # Rename time, OLR, precipitation variables
    ds = ds.rename({
        time_coordname: 'time', 
        x_coordname: 'lon',
        y_coordname: 'lat',
        olr_varname: 'olr', 
        pcp_varname: 'precipitation',
    })
    # Replace OLR & precipitation attributes
    olr_attr = {
        'long_name': 'Top of atmosphere outgoing longwave radiation',
        'units': 'W m-2',
    }
    pcp_attr = {
        'long_name': 'Surface precipitation rate',
        'units': 'mm h-1',
    }
    ds['olr'].attrs = olr_attr
    ds['precipitation'].attrs = pcp_attr
    # For OBS, keep Tb
    if runname == 'OBS':
        ds = ds.rename({'olr':'Tb'})
        ds['Tb'].attrs = {
            'long_name': 'Brightness temperature',
            'units': 'K',
        }

    # Replace global attributes
    gattrs = {
        'Title': f'{PHASE} {runname} OLR & precipitation',
        'phase': f'{PHASE}',
        'source': f'{runname}',
        'Created_on': time.ctime(time.time()),
    }
    ds.attrs = gattrs
    
    # Output datetime string
    out_datetime_str = ds['time'].dt.strftime('%Y%m%d%H').item()
    # Double check output datetime string from the file is the same with the original filename
    if out_datetime_str != fn_datetime:
        print(f'ERROR: converted time from file ({out_datetime_str}) is NOT the same with filename ({fn_datetime})')
        print(f'Input file: {filename}')
        sys.exit(f'Code will exit now.')

    # Output filename
    out_filename = f'{out_dir}{out_basename}{out_datetime_str}.nc'
    
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds.data_vars}

    # Write to netcdf file
    ds.to_netcdf(path=out_filename, mode="w", format="NETCDF4", unlimited_dims='time', encoding=encoding)
    print(f"{out_filename}")

    # import pdb; pdb.set_trace()
    return


if __name__ == "__main__":
    
    # Get inputs from command line
    config_file = sys.argv[1]
    
    print(f'Start time: {time.ctime(time.time())}')

    # Get runname and PHASE from config_file name
    parts = config_file.split("/")
    config_file_basename = parts[-1]
    # Config filname format: config_dyamond_PHASE_runname.yml
    runname = config_file_basename.split("_")[-1].split(".")[0]
    PHASE = config_file_basename.split("_")[-2].capitalize()
    print(f'{PHASE} {runname}')

    if PHASE == 'Summer':
        start_datetime = '2016-08-01 00:00:00'
    elif PHASE == 'Winter':
        start_datetime = '2020-01-20 00:00:00'

    # Output directory and basename
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/OLR_Precipitation/{PHASE}/{runname}/'
    out_basename = f'olr_pcp_{PHASE}_{runname}_'
    os.makedirs(out_dir, exist_ok=True)

    # Reference grid
    ref_grid = '/pscratch/sd/f/feng045/DYAMOND/maps/IMERG_landmask_180W-180E_60S-60N.nc'
    
    # Parallel setup
    run_parallel = 1
    n_workers = 128

    # Get inputs from configuration file
    config = load_config(config_file)
    in_dir = config['clouddata_path']
    databasename = config['databasename']

    # Find all input files
    in_files = sorted(glob.glob(f'{in_dir}{databasename}*nc'))
    nfiles = len(in_files)
    print(f'Number of input files: {nfiles}')

    # Start Dask cluster
    if run_parallel == 1:
        # Set Dask temporary directory for workers
        dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        # Local cluster
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)

    results = []
    # Loop over each file
    for ifile in in_files:
        if run_parallel == 0:
            result = harmonize_file(ifile)
        elif run_parallel == 1:
            result = dask.delayed(harmonize_file)(ifile)
        results.append(result)

    if run_parallel == 1:
        # Trigger dask computation
        final_result = dask.compute(*results)
        wait(final_result)

        # Close the Dask cluster
        client.close()
        cluster.close()

    print(f'End time: {time.ctime(time.time())}')