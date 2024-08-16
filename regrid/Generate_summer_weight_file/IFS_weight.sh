#!/bin/bash

module load cdo

# mars_1.nc is a file of IFS-4km model

cdo -P 16 --cellsearchmethod spherepart -genycon,0.10_grid.nc mars_1.nc ECMWF-4km_0.10_grid_wghts.nc
