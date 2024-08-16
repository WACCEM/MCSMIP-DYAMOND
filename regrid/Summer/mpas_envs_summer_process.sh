#!/bin/bash
#SBATCH --job-name=mpas_ens_process
#SBATCH --partition=shared
#SBATCH --account=bb1153
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --time=48:00:00
#SBATCH --output=mpas_ens_process.%j.out
#diag.2016-09-09_01.30.00.nc
module load cdo
module load nco
out_dir='/home/b/b382080/store-data/mpas/'
remap_dir='/home/b/b382080/store-data/regrid/grid/Summer/'
data_dir='/fastdata/ka1081/DYAMOND/data/summer_data/MPAS-3.75km/'
mins=('00' '15' '30' '45')

variable_list=('q2')
mon=(8 9)
#diag.2016-09-01_00.00.00.nc
for m in ${mon[@]};do
    for d in {05..31};do
        for h in {00..24};do
             for min in ${mins[@]};do
                 echo 2016-0${m}-${d}-${h}-${min}
                 cdo -P 8 -f nc4 -remap,${remap_dir}/0.10_grid.nc,${remap_dir}/MPAS-3.75km_0.10_grid_wghts.nc -setgrid,${remap_dir}/mpas_setgrid.nc -selname,${variable_list} ${data_dir}/diag.2016-0${m}-${d}_${h}.${min}.00.nc ${out_dir}/${variable_list}_diag_0.10deg.2016-0${m}-${d}_${h}.${min}.00.nc
             done
        done
    done
done 

#Since there is no time axis, the file is linked with -cat
#Setting the MPAS model variable time axis, 2016-08-01 00:00:00 at 15min intervals, and taking hourly mean
cdo -hourmean -sellonlatbox,-180,180,-60,60 -settaxis,2016-08-01,00:00:00,15min -cat ${out_dir}/${variable_list}_diag_0.10deg* ${out_dir}/${variable_list}_diag_0.10deg_1hr_-60_60.nc
