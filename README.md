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