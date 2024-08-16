#!/bin/bash
#SBATCH --job-name=nicam
#SBATCH --partition=shared
#SBATCH --account=bb1153
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --time=48:00:00
#SBATCH --output=nicam.%j.out
module load cdo/
module load nco
monthdays=(0801 0802 0803 0804 0805 0806 0807 0808 0809 0810 0811 0812 0813 0814 0815 0816 0817 0818 0819 0820 0821 0822 0823 0824 0825 0826 0827 0828 0829 0830 0831 0901 0902 0903 0904 0905 0906 0907 0908 0909)

data_dir='/home/b/b382080/dyamond_summer/NICAM-3.5km/'
remap_dir='/home/b/b382080/store-data/regrid/grid/Summer/'
out_dir='/home/b/b382080/store-data/nicam/'
vars=('ms_qv' 'ms_rh' 'ms_u' 'ms_v')

for var in ${vars[@]};do
    mkdir ${out_dir}/summer-${var}
    for dd in ${monthdays[@]};do
        echo $d
        cdo -remap,${remap_dir}/0.10_grid.nc,${remap_dir}/NICAM-3.5km_0.10_grid_wghts.nc ${data_dir}/2016${dd}*/${var}.nc ${out_dir}/summer-${var}/${var}_2016${dd}_0.1deg.nc
    done
    cdo -sellonlatbox,-180,180,-60,60 -hourmean -mergetime ${out_dir}/summer-${var}/${var}_2016*_0.1deg.nc ${out_dir}/summer-${var}/${var}_1hr_nicam_0.1deg.nc
done
