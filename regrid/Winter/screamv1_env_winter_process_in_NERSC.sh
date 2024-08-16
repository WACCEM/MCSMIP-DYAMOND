#!/bin/bash
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --tasks-per-node=100
#SBATCH --output=winter_envs_process.%j.out
# Begin of section with executable commands

# Start the virtual environment and use the cdo
conda activate song
model='SCREAMv1'
remap_dir='/global/u2/s/sjy/sjy-store/regrid/grid/Winter/'

vars=('surf_evap' 'surf_sens_flux' 'ps' 'T_2m' 'qv_2m' 'wind_speed_10m')
cd /pscratch/sd/f/feng045/DYAMOND2_SCREAMv1/envs/

for month in {01..02};do
	for day in {01..31};do
		echo ${month}-${day}
		cdo -sellonlatbox,-180,180,-60,60 -remapcon,${remap_dir}/grid_0.1x0.1 -setgrid,${remap_dir}/${model}/grid.nc -selname,surf_evap,surf_sens_flux,ps,T_2m,qv_2m,wind_speed_10m ../output.scream.SurfVars.INSTANT.nmins_x15.2020-${month}-${day}-*.nc temp1.nc
		cdo -splitname,swap temp1.nc _1hr_SCREAMv1_2020-${month}-${day}
		rm -f temp1.nc
	done
done

for var in ${vars[@]};do
	cdo -hourmean -mergetime ${var}_1hr_SCREAMv1* ${var}_1hr_SCREAMv1_20200120-20200228_60S-60N.nc
done
                







