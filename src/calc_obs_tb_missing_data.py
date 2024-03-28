"""
Calculate valid Tb counts and save output to a netCDF file.
"""
import sys
import xarray as xr
import time

if __name__ == "__main__":

    PHASE = sys.argv[1]
    runname = sys.argv[2]

    # Input data
    # tb_dir = f'/pscratch/sd/f/feng045/DYAMOND/OLR_Precipitation_combined_regrid/'
    tb_dir = f'/pscratch/sd/f/feng045/DYAMOND/OLR_Precipitation_combined/'
    tb_file = f'{tb_dir}olr_pcp_{PHASE}_{runname}.nc'
    # Output data
    out_dir = tb_dir
    out_filename = f'{out_dir}{PHASE}_{runname}_validcount.nc'

    # Read Tb data
    ds = xr.open_dataset(tb_file)
    lon = ds['lon']
    lat = ds['lat']
    ntimes = ds.sizes['time']
    time0 = ds.time.isel(time=0)

    # Make a mask for valid data (> 0) for all times == 00 min and 30 min, sum over time to get counts
    valid_count = (ds['Tb'] > 0).sum(dim='time')

    #-------------------------------------------------------------------------
    # Write output file
    #-------------------------------------------------------------------------
    # Define xarray output dataset
    vars_dict = {
        'valid_count': (['lat', 'lon'], valid_count.data),
        # 'ntimes': (['time'], ntimes),
    }
    coords_dict = {
        'lon': (['lon'], lon.data, lon.attrs),
        'lat': (['lat'], lat.data, lat.attrs),
    }
    gattrs_dict = {
        'title': f'{PHASE} {runname} valid Tb data counts',
        'total_ntimes': ntimes,
        'contact':'Zhe Feng, zhe.feng@pnnl.gov',
        'created_on':time.ctime(time.time()),
    }
    dsout = xr.Dataset(vars_dict, coords=coords_dict, attrs=gattrs_dict)
    dsout['valid_count'].attrs['long_name'] = 'Valid data count'
    dsout['valid_count'].attrs['units'] = 'count'

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}
    dsout.to_netcdf(path=out_filename, mode='w', format='NETCDF4', encoding=encoding)
    print(f'Output saved: {out_filename}')
