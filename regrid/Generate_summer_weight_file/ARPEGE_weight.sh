#!/bin/bash

module load cdo
#ARPEGE/ARPNH2D201608082330 is a nc file for ARPEGE model

cdo -P 16 --cellsearchmethod spherepart genycon,/work/ka1081/Hackathon/GrossStats/0.10_grid.nc -setgrid,griddes.arpege1 -setgridtype,regular ARPEGE/ARPNH2D201608082330 ARPEGE-NH_wghts.nc
