"""
Spatially average 2D variable center at each MCS track and save the output to a netCDF file
"""
import os, sys
import time
import xarray as xr
import numpy as np

if __name__ == "__main__":

    PHASE = sys.argv[1]
    runname = sys.argv[2]
    tracker = sys.argv[3]
    varname = sys.argv[4]

    stats_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_stats/{PHASE}/{tracker}/'
    out_dir = stats_dir
    in_file = f"{stats_dir}mcs_tracks_{PHASE}_{runname}_{varname}_2d.nc"
    out_filename = f'{out_dir}mcs_tracks_{PHASE}_{runname}_{varname}_1d.nc'

    # Define averaging radius in [degrees]
    box_radii = [0.5, 1.0, 1.5, 2.5]
    radius_coord = xr.DataArray(box_radii, dims='radius', coords={'radius': box_radii}, attrs={'long_name': 'Average radius', 'units': 'degree'})

    # Data grid spacing [degree]
    if runname == 'OBS':
        dx = 0.25
    else:
        dx = 0.1

    # Read input data
    ds = xr.open_dataset(in_file)
    # # Get coordinates in degree
    # x = ds['x'] * dx
    # y = ds['y'] * dx
    
    # Create a mask array
    mask_shape = (ds.sizes['y'], ds.sizes['x'])
    # Define the center and radius of the circle
    center = np.array(mask_shape) // 2  # Center of the array
    # Create a grid of coordinates
    ym, xm = np.ogrid[:mask_shape[0], :mask_shape[1]]
    
    # Loop over each radius for averaging
    ds_save = []
    for i in range(len(box_radii)):
        # # Find indices for x, y dimensions
        # xid = np.where((x >= -1*box_radii[i]) & (x <= box_radii[i]))[0]
        # yid = np.where((y >= -1*box_radii[i]) & (y <= box_radii[i]))[0]
        # Subset spatial area, then average
        # ds_avg = ds.isel(x=slice(min(xid), max(xid)), y=slice(min(yid), max(yid))).mean(dim=('y', 'x'), keep_attrs=True)

        # Convert radius from [degree] to number of grid points
        ngrids_radii = box_radii[i] / dx
        # Create an array for the mask
        array = np.zeros(mask_shape)
        # Use the distance formula to create the circular mask
        mask = ((xm - center[1]) ** 2 + (ym - center[0]) ** 2) <= ngrids_radii ** 2
        # Apply the circular mask to the array
        array[mask] = 1
        # Convert mask to DataArray
        mask_da = xr.DataArray(array, coords={'y':ds['y'], 'x':ds['x']}, dims=('y','x'))

        # Apply mask to DataSet, then average
        ds_avg = ds.where(mask_da).mean(dim=('y', 'x'), keep_attrs=True)
        # import pdb; pdb.set_trace()
        # ds_avg['radius'] = radius_coord
        ds_avg = ds_avg.assign_coords({'radius':radius_coord[i]})
        ds_save.append(ds_avg)

    # Concatenate datasets along the new dimension 'radius'
    dsout = xr.concat(ds_save, dim='radius')

    # import pdb; pdb.set_trace()
    # Replace global attributes
    gattr_dict = {
        'Title': 'Mean 2D environments for MCS',
        # 'spatial_avg_width_degree': box_radii * 2,
        'Institution': 'Pacific Northwest National Laboratoy',
        'Contact': 'zhe.feng@pnnl.gov',
        'Created_on': time.ctime(time.time()),
    }
    dsout.attrs = gattr_dict

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}
    dsout.to_netcdf(path=out_filename, mode='w', format='NETCDF4', unlimited_dims='tracks', encoding=encoding)
    print('Output saved as: ', out_filename)

    # import pdb; pdb.set_trace()