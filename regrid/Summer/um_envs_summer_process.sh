#!/bin/bash
#SBATCH --job-name=UM
#SBATCH --partition=shared
#SBATCH --account=bb1153
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=120
#SBATCH --time=48:00:00
#SBATCH --output=UM_summer_data_process.%j.out
module load cdo
module load nco
remap_dir='/home/b/b382080/store-data/regrid/grid/Summer/'
out_dir='/home/b/b382080/store-data/um/'
data_dir='/fastdata/ka1081/DYAMOND/data/summer_data/UM-5km/'

monthdays=(0801 0802 0803 0804 0805 0806 0807 0808 0809 0810 0811 0812 0813 0814 0815 0816 0817 0818 0819 0820 0821 0822 0823 0824 0825 0826 0827 0828 0829 0830 0831 0901 0902 0903 0904 0905 0906 0907 0908 0909)
vars=('ps' 'huss')

echo 'remapping UM fields to 0.1x01.deg lat lon grid'
for var in ${vars[@]};do
    mkdir ${out_dir}/summer-${var}
    for d in ${monthdays[@]} ; do
        cdo -P 4 -remap,${remap_dir}/0.10_grid.nc,${remap_dir}/UM-5km_0.10_grid_wghts.nc ${data_dir}/${var}/${var}_15min_HadGEM3-GA71_N2560_20160${d}.nc ${out_dir}/summer-${var}/${var}_1hr_HadGEM3-GA71_N2560_20160${d}_0.10deg.nc
    done
    cdo -mergetime ${out_dir}/summer-${var}/${var}_1hr_HadGEM3-GA71_N2560_20160*_0.10deg.nc ${out_dir}/summer-${var}/${var}_1hr_HadGEM3-GA71_N2560_0.10deg.nc
done

#cdo -mergetime ${out_dir}/summer-${var}/${var}_1hr_HadGEM3-GA71_N2560_20160*_0.10deg.nc ${out_dir}/summer-${var}/${var}_1hr_HadGEM3-GA71_N2560_0.10deg.nc


