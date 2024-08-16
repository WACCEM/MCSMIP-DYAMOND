#!/bin/bash
#SBATCH --job-name=nicam
#SBATCH --partition=shared
#SBATCH --account=bb1153
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --time=48:00:00
#SBATCH --output=sam_summer_pw_sp.%j.out
module load cdo
module load nco
out_dir='/home/b/b382080/store-data/sam/'
remap_dir='/home/b/b382080/store-data/regrid/grid/Summer/'
data_dir='/fastdata/ka1081/DYAMOND/data/summer_data/SAM-4km/OUT_2D/'
var_list=("Q2m")

## remapping
echo 'remapping SAM fields to 0.1x01.deg lat lon grid'
for i in ${var_list[@]}; do
    echo 'processing '${i}

    #Since there is no time axis, the file is linked with -cat
    cdo -cat ${data_dir}/DYAMOND_9216x4608x74_7.5s*.${i}.2D.nc ${out_dir}/SAM_${i}.nc
    #Setting the MPAS model variable time axis, 2016-08-01 00:30:00 at 30min intervals
    cdo -r -P 8 -settaxis,2016-08-01,00:30:00,30m -remap,${remap_dir}/0.10_grid.nc,${remap_dir}/SAM_0.10_grid_wghts.nc ${out_dir}/SAM_${i}.nc ${out_dir}/SAM_${i}_0.10deg.nc
    cdo -sellonlatbox,-180,180,-60,60 -hourmean  ${out_dir}/SAM_${i}_0.10deg.nc  ${out_dir}/SAM_1hr_${i}_0.10deg_-60_60.nc  
done

