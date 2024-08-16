#!/bin/bash

module load cdo

remap_dir='/home/b/b382080/store-data/regrid/grid/Summer/'

#diag.2016-08-01_00.15.00.nc is a file of MPAS model

cdo -P 16 --cellsearchmethod spherepart -genycon,0.10_grid.nc -setgrid,${remap_dir}/MPAS_setgrid.nc diag.2016-08-01_00.15.00.nc MPAS-3.75km_0.10_grid_wghts.nc
