"""
This script regrids DPR swath data to the global IMERG grid of 0.1x0.1 degrees.

For each granule, two output files are produced:
- *src-grid.nc: netCDF file with granule info at native grid, but restructured so it can be read with xarray
- *regridded.nc: netCDF file containing the granule regridded to regular global IMERG grid 


Email: kukulies@ucar.edu

"""
import os
import sys 
from pathlib import Path 
import xarray as xr 
import numpy as np
import scipy
from datetime import datetime, timedelta
import pandas as pd
import xesmf as xe

# form pansat library
import pansat
from pansat.environment import get_index
from pansat import TimeRange
from pansat.geometry import LonLatRect 
from pansat.granule import merge_granules
from pansat.catalog import Index
from pansat.products.satellite.gpm import l2a_gpm_dpr

# from pyflextrkr library
from pyflextrkr.ft_regrid_func import  make_grid4regridder, make_weight_file, get_latlon_bounds_1d, get_latlon_bounds_2d
from pyflextrkr.regrid_tracking_mask import regrid_tracking_mask

print('start processing DPR data..', flush = True)
index = get_index(l2a_gpm_dpr)

# location of granule data files
path = Path(sys.argv[1])
# where to save the regridded data 
outdir= Path(sys.argv[2])

# get meta data for granules (start and end time required)
start_time = sys.argv[3]
end_time   = sys.argv[4]

# find granules that fall in a specific period
granules   = index.find(TimeRange(start_time, end_time))
granules = merge_granules(granules)
print(len(granules), 'found for period.', flush = True)
print(granules[0], flush = True)

############## SETTING GENERIC PARAMETERS FOR REGRIDDING #####################################

config = dict()
config['regrid_method'] = 'conservative'
config['regrid_input'] = True 
# directory of example file that is the target grid (here: IMERG)
config['gridfile_dst'] = sys.argv[5]
config['x_coordname_src'] =  'longitude'
config['y_coordname_src'] =  'latitude'
config['x_coordname_dst'] =  'lon'
config['y_coordname_dst'] =  'lat'
config['write_native'] =  False

###########################################################################################

# loop through each granule 
for granule in granules:
    granule.file_record.local_path = path / granule.file_record.filename
    granule_ds = granule.open()[{"frequencies": -2}]
    granule_ds['surface_precip'] = granule_ds.surface_precip.where(granule_ds.surface_precip > 0, 0 )
    granule_name = str(granule)[18:77]
    lats = granule_ds.latitude
    lons = granule_ds.longitude
    
    ### get average granule time ### 
    time_avg = granule_ds.scan_time.mean()  
    # get next half hour to that average
    timestr = str(time_avg.dt.round('30min').data)[0:-10]
    time = pd.to_datetime(time_avg.dt.round('30min').data )
    
    print(granule_name, time, timestr, flush = True)
    
    # unique output filename 
    outfilename = outdir /  str( granule_name + '_'+ timestr +'_regridded.nc')

    # check if file has already been processed
    if os.path.isfile(outfilename) is False:
            # create source file 
            fname = outdir /  (granule_name + '_' + timestr +  '_src-grid.nc')
            # save source grid
            granule_ds.to_netcdf(fname)
            gridfile_src = fname
            #config['weight_filename']  = outdir / str('weightfile_' + granule_name + '.nc')
            
            # regridding steps 
            grid_src, grid_dst = make_grid4regridder(gridfile_src, config)
            print('got grid information, start regridding..', flush = True)

            try:
                    regridder = xe.Regridder(grid_src, grid_dst, method='conservative')
                    # apply regridder
                    data_regridded = regridder(granule_ds.surface_precip.data)

                    # mask data outside of granule 
                    granule_data = granule_ds.precip_type.where(granule_ds.precip_type < 0  , 1000 )
                    mask = regridder(granule_data.data)
                    data_regridded[mask == 0 ] = np.nan

                    ##### SAVE OUTPUT  #############
                    # Output coordinates
                    output = np.expand_dims(data_regridded, 0)
                    x_coord = grid_dst['lon']
                    y_coord = grid_dst['lat']
                    x_coord = xr.DataArray(x_coord, dims=('lon'))
                    y_coord = xr.DataArray(y_coord, dims=('lat'))

                    # Make output DataSet
                    coords = {'time': [time] , 'lat': y_coord,  'lon': x_coord}
                    dims = ['time', 'lat', 'lon']
                    ds_out = xr.Dataset( data_vars =  {'precipitation_dpr': ( dims, output  )} , coords = coords ) 

                    comp = dict(zlib=True)
                    encoding = {var: comp for var in ds_out.data_vars}
                    # Write to netCDF file 
                    ds_out.to_netcdf(path=outfilename,mode="w",format="NETCDF4", unlimited_dims="time",encoding=encoding,)

                    print(outfilename, ' written. ', flush = True)
                    ds_out.close()
                    granule_ds.close()
            except:
                    print(outfilename, 'failed to regrid,', flush = True )
                    continue

    else:
            print(outfilename, 'already processed.', flush = True )

    


