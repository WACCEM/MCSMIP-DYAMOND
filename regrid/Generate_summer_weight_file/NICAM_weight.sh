#!/bin/bash

module load cdo

# sa_cldi.nc is a file of NICAM model
cdo -P 16 --cellsearchmethod spherepart genycon,0.10_grid.nc -seltimestep,1 sa_cldi.nc NICAM-3.5km_0.10_grid_wghts.nc
