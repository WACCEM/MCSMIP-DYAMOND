#!/bin/bash
#SBATCH --job-name=ifs
#SBATCH --partition=shared
#SBATCH --account=bb1153
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=150
#SBATCH --time=72:00:00
#SBATCH --output=ifs_summer_envs_processing.%j.out
module load cdo
module load nco
data_dir='/fastdata/ka1081/DYAMOND/data/summer_data/IFS-4km/'
out_dir='/home/b/b382080/store-data/ifs/'
remap_dir='/home/b/b382080/store-data/regrid/grid/Summer/'

variables_name=('10v' '10u')
for variable_name in ${variables_name[@]}; do
    echo 'processing file ' ${variable_name}
    mkdir ${out_dir}/summer-${variable_name}
    
    for var in {0..960}; do
      #the file name of IFS model is xxxx.[0-960]  
      #the number of xxx.70 indicate the 70 hour
      echo 'processing file ' ${var}
      
      cdo --eccodes select,name=${variable_name} ${data_dir}/mars_out.${var} ${out_dir}/summer-${variable_name}/mars_${var}.grb
      
      cdo -R -f nc copy ${out_dir}/summer-${variable_name}/mars_${var}.grb ${out_dir}/summer-${variable_name}/mars_${var}.nc
      ## Note that you must change the variable name, not changing it will only get the variable number (change it according to the file description)
      cdo remap,${remap_dir}/0.10_grid.nc,${remap_dir}/ECMWF-4km_0.10_grid_wghts.nc -chname,var166,v10m,var165,u10m  ${out_dir}/summer-${variable_name}/mars_${var}.nc ${out_dir}/summer-${variable_name}/mars_${var}_0.10deg.nc
      
      rm ${out_dir}/summer-${variable_name}/mars_${var}.nc ${out_dir}/summer-${variable_name}/mars_${var}.grb
   done
   cdo -hourmean -sellonlatbox,-180,180,-60,60 -mergetime ${out_dir}/summer-${variable_name}/mars_*_0.10deg.nc  ${out_dir}/summer-${variable_name}/${variable_name}_1hr_0.10deg_-60_60.nc
done



