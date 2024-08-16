#!/bin/bash

module load cdo

#SAM_PRECdeacc.nc is a file of SAM model
cdo -P 16 --cellsearchmethod spherepart genycon,0.10_grid.nc -seltimestep,1 SAM_${i}.nc SAM_0.10_grid_wghts.nc


