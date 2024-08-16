#!/bin/bash

module load cdo

#ps_15min_HadGEM3-GA71_N2560_20160801.nc is a file of UM model
cdo -P 16 --cellsearchmethod spherepart genycon,0.10_grid.nc ps_15min_HadGEM3-GA71_N2560_20160801.nc UM-5km_0.10_grid_wghts.nc
