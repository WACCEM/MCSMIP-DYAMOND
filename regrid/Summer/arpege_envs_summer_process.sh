#! /bin/bash

module load cdo
module load nco
#Loads the specified environment for converting to grib files for ARPEGE model
export GRIB_DEFINITION_PATH=/home/b/b382080/dyamond_summer/ARPEGE-NH-2.5km/GRIB_DEFINITION_PATH

data_dir='/home/b/b382080/store-data/arpege'
out_dir='/work/bb1153/from_Mistral/bb1153/DYAMOND_USER_DATA/Hackathon/GrossStats'

for day in {20160801..20160909}; do
   echo 'processing file ' ${day}
   cd /home/b/b382080/dyamond_summer/ARPEGE-NH-2.5km/${day}
       for hh in ARPNH2D${day}* ; do
#         Convert the file to the grib format
          /home/b/b382080/dyamond_summer/ARPEGE-NH-2.5km/documentation_and_scripts/gribmf2ecmwf ${hh} ${tmp_dir}/${hh}
          cdo -f nc4 remap,${out_dir}/0.10_grid.nc,${out_dir}/ARPEGE-2.5km_0.10_grid_wghts.nc -setgrid,${out_dir}/griddes.arpege1 -setgridtype,regular ${tmp_dir}/${hh} ${tmp_dir}/${hh}.nc
       done
done

