# **MCSMIP DYAMOND Analysis**


---
This repository contains analysis codes and Jupyter Notebooks for the MCSMIP (MCS tracking Method Intercomparison Project) for DYAMOND simulations.

The project details are available in this [GoogleDoc](https://docs.google.com/document/d/1vdSJmqrmpch6Ck68Jbt00WF6RG2SBWxLJ6zmocKb9Y4/edit?usp=sharing).


## Batch Processing in Slurm
---

All processing codes are in the [src directory](https://github.com/WACCEM/MCSMIP-DYAMOND/tree/main/src). Most codes are parallelized with Dask that can be run on a single node, which typically finishes in 10-15 min.

This script generates specific processing tasks and  submits to slurm:

`python make_mcs_stats_joblib.py ${PHASE} ${tracker}`

Optionally, this Bash script can run all jobs for a single DYAMOND source and a single tracker:

`run_alljobs_1source.sh`

Below are more details for different sets of processing codes.


## Regrid DYAMOND Data
---
CDO scripts used to regrid raw DYAMOND data to lat/lon grids are in the [regrid directory](https://github.com/WACCEM/MCSMIP-DYAMOND/tree/main/regrid).


## Standardize Datasets
---

The `${}` are command line inputs, examples:

> ${PHASE}: 'Summer' or 'Winter'
> 
> ${runname}: 'OBS', 'NICAM', 'SCREAM'
> 
> ${tracker}: 'PyFLEXTRKR', 'TOOCAN'
> 
> ${env_varname}: 'intqv', 't2m'
> 
> ${start_date}: '2016-08-01T00'
> 
> ${end_date}: '2020-03-01T00'

--

* **Combine PyFLEXTRKR MCS mask files to a single file:**

`python make_mcs_maskfile_singlefile.py ${PHASE} ${runname}`

* **Standardize other tracker mask files:**

`python unify_mask_files.py ${PHASE} ${runname} ${tracker}`

* **Standarize DYAMOND environmental files:**

`python unify_env_files.py ${PHASE} ${runname} ${env_varname}`

## Download DPR swath data 

`python download_dpr.py ${outdir} ${start_date} ${end_date}`

> ${outdir}: directory where to store the downloaded data 
> 
> ${start_date}: start date and time, e.g. 2020-02-01T00:00:00
> 
> ${end_date}: end date and time, e.g. 2020-02-28T23:00:00

## Visualization

---
* **Tb + precipitation + MCS mask animation for any DYAMOND sources and MCS trackers:**

`python make_mcs_quicklook_animation.py ${PHASE}`

* **MCS swath masks and counts for any DYAMOND sources and MCS trackers:**

`python make_mcs_maskswath_plots.py ${PHASE}`


## Regrid data to ERA5 grid:
---
* **Regrid DYAMOND environment data to ERA5:**

`python regrid_envs2era5.py ${PHASE} ${runname} ${env_varname}`

* **Regrid Tb & precipitation data to ERA5:**

`python regrid_tbpcp2era5.py ${PHASE} ${runname}`

* **Regrid MCS mask to ERA5:**

`python regrid_mcsmask2era5.py ${PHASE} ${runname} ${tracker}`

## Regrid GPM DPR swath data to IMERG grid:

`python regridding_dpr.py ${data_path} ${outdir} ${start_date} ${end_date} ${target_grid}`

> 
> ${data_path}: directory that contains the downloaded DPR data files
> 
> ${outdir}: directory where to store the regridded files  
> 
> ${start_date}: start date and time, e.g. 2020-02-02T00:00:00
> 
> ${end_date}: end date and time, e.g. 2020-02-28T23:00:00
> 
> ${target_grid}: directory of a file with the target grid, e.g. data/olr_pcp_Winter_OBS_2020022612.nc


## Calculate Statistics

---
* **Calculate MCS track statistics from mask file:**

`python make_mcs_stats_from_maskfile.py ${config_file} ${tracker}`

* **Calculate time-mean MCS frequency & precipitation map:**

`python calc_tbpf_mcs_rainmap_mcsmip.py ${config_file} ${tracker} ${start_date} ${end_date}`

* **Global mean precipitation time series:**

`python avg_global_rain_timeseries.py ${PHASE} ${runname}`

* **Global mean environment time series:**

`python avg_global_env_map_timeseries.py ${PHASE} ${runname} ${env_varname} ${start_date} ${end_date}`

* **Tb and rain rate histogram:**

`python calc_tb_rainrate_pdf_byregion.py ${PHASE} ${runname} ${tracker}`

* **Extract 2D environments for MCS tracks:**

`python extract_mcs_2d_env.py ${PHASE} ${runname} ${tracker} ${env_varname}`

* **Average 2D environments for MCS tracks:**

`python avg_mcs_track_env_space.py ${PHASE} ${runname} ${tracker} ${env_varname}`

* **Calculate mean MCS precipitation bin by environment:**

`python calc_mcs_pcp_envs_pairs.py ${PHASE} ${runname} ${tracker} ${env_varname} ${start_date} ${end_date}`

## Analysis 
---
Analysis and plotting are in the [Notebooks directory](https://github.com/WACCEM/MCSMIP-DYAMOND/tree/main/Notebooks). More details will be added later.