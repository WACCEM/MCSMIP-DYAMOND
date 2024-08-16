#!/bin/bash
#The raw data comes from the Levante server, which is the script for processing GRIST model in Levante
#The following paths are paths in Levante
module load cdo
module laod nco

# Choose the model you need data path
data_dir=/fastdata/ka1081/DYAMOND/data/winter_data/CAMS/GRIST-5km/DW-ATM/atmos/15min/

#Variable names (according to the name of the given model)
vars=('tas')

#This is the DYAMONAD winter model
#Choose the model you need
model=('GRIST')

out_dir=/home/b/b382080/store-data/${model}/winter-${var}
grid_dir=/home/b/b382080/store-data/regrid/grid/Winter/

cd ${data_dir}
for var in ${vars[@]}
do
	mkdir ${out_dir}
        echo ${var}-start
        cdo -mergetime ${var}/r1i1p1f1/2d/gn/${var}_15min_GRIST-5km_DW-ATM_r1i1p1f1_2d_gn_2020* ${out_dir}/temp0.nc
       
        cdo -remapcon,${grid_dir}/grid_0.1x0.1 -setgrid,${grid_dir}/${model}/grid.nc ${out_dir}/temp0.nc ${out_dir}/temp1.nc
  
        cdo -hourmean -sellonlatbox,-180,180,-60,60 ${out_dir}/temp2.nc ${out_dir}/${var}_15min_GRIST-5km_DW-ATM_r1i1p1f1_2d_gn_20200120-20200228_60S-60N.nc

        rm -f ~/store-data/${model}/winter-${var}/temp*
done

