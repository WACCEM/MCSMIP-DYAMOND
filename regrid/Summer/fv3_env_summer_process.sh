#!/bin/bash
#SBATCH --job-name=noaa_data
#SBATCH --partition=shared
#SBATCH --account=bb1153
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --time=48:00:00
#SBATCH --output=noaa_data.%j.out

module load cdo
module load nco
out_dir='/home/b/b382080/store-data/fv3/'
data_dir='/fastdata/ka1081/DYAMOND/data/summer_data/FV3-3.25km/'
remap_dir='/home/b/b382080/store-data/regrid/grid/Summer'
tmp_dir='/work/bb1153/from_Mistral/bb1153/DYAMOND_USER_DATA/Hackathon/GrossStats'

days=(2016080100 2016081100 2016082100 2016083100)

#The cubed-sphere is made of 6 tiles. Therefore, each native model variable is stored in 6 separate files. Each tile is a 3072 by 3072 square mesh.
#atmos_static.tile?.nc and grid_spec.tile?.nc
#The static data which store the horizontal and vertical grid information

# Generate weight file
module swap cdo cdo/1.9.5-magicsxx-gcc64
var_list=("intqv")
echo 'creating remapping weights from FV3 cubed spere to 0.1x01.deg lat lon grid'
for N in {1..6}; do
     cdo import_fv3grid ${data_dir}/2016080100/grid_spec.tile$N.nc ${out_dir}/gridspec.tile$N.nc
done
cdo collgrid ${out_dir}/gridspec.tile?.nc ${out_dir}/gridspec.nc
cdo -P 12 genycon,${remap_dir}/0.10_grid.nc -setgrid,${out_dir}/gridspec.nc -collgrid,gridtype=unstructured ${data_dir}/2016080100/intqv_15min.tile?.nc  ${out_dir}/fv3_weights_to_0.10deg.nc

## remapping
echo 'remapping FV3 fields to 0.1x01.deg lat lon grid'

for var in ${var_list[@]}; do
    echo 'remapping ' ${var}
    mkdir ${out_dir}/summer_${var}
    for dd in ${days[@]}; do
          cdo -P 12 -remap,${remap_dir}/0.10_grid.nc,${remap_dir}/fv3_weights_to_0.10deg.nc -setgrid,${remap_dir}/gridspec.nc -collgrid,gridtype=unstructured ${data_dir}/${dd}/${var}_15min.tile?.nc ${out_dir}/summer_${var}/FV3_${var}_${dd}.nc
    done
    cdo -hourmean -sellonlatbox,-180,180,-60,60 -mergetime ${out_dir}/summer_${var}/FV3_${var}*.nc ${out_dir}/summer_${var}/fv3_${var}_0.10deg_hourmean.nc
done
