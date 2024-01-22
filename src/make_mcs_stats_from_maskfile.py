"""
Calculate MCS Tb and precipitation statistics from mask files and OLR/precipitation files for a given tracker,
and save the statistics to a netCDF file similar to the format used by PyFLEXTRKR.
"""
__author__ = "Zhe.Feng@pnnl.gov"

import numpy as np
import xarray as xr
import pandas as pd
import yaml
import time, os, sys
from scipy.ndimage import label
import dask
from dask.distributed import Client, LocalCluster, wait
from pyflextrkr.ftfunctions import olr_to_tb, sort_renumber
from pyflextrkr.matchtbpf_func import calc_pf_stats
# import matplotlib as mpl
# import matplotlib.pyplot as plt

#-----------------------------------------------------------------------
def find_track_time_indices(list_tracks, track_number):
    """
    Find time indices in the list for a given track number
    """
    # Initialize a list to store the indices of lists containing the track number
    indices_with_track = []
    # Iterate through the lists and find the indices where the track number is present
    for i, lst in enumerate(list_tracks):
        if track_number in lst:
            indices_with_track.append(i)

    return indices_with_track

#-----------------------------------------------------------------------
def featurestats_singlefile(
        olrpcp_filename,
        file_mcsnumber,
        imcs_mask,
        config,
):
    """
    Calculate Tb and PF statistics for a single file

    Args:
        olrpcp_filename: string
            Filename for the OLR & precipitation data.
        file_mcsnumber: array-like
            MCS track numbers in this file.
        imcs_mask: array-like
            MCS mask array.
        config: dictionary
            Dictionary containing the config.

    Returns:
        out_dict: dictionary
            Dictionary containing output variable arrays.
        out_dict_attrs: dictionary
            Dictionary containing output variable attributes.
        var_names_2d: dictionary
            Dictionary containing 2D variable names.
    """    
    olr2tb = config.get('olr2tb', None)
    olr_varname = config.get('olr_varname')
    tb_varname = config.get('tb_varname')
    pcp_varname = config['pcp_varname']
    time_dimname = config['time_dimname']
    landmask_filename = config.get("landmask_filename", "")
    landmask_varname = config.get("landmask_varname", "")
    landfrac_thresh = config.get("landfrac_thresh", 0)

    # Read landmask file
    dslm = xr.open_dataset(landmask_filename)
    landmask = dslm[landmask_varname].squeeze().data
    lon1d = dslm['lon'].data
    lat1d = dslm['lat'].data
    # Make 2D lat/lon
    lon, lat = np.meshgrid(lon1d, lat1d)
    ny, nx = lon.shape

    # Read cloudid file
    if os.path.isfile(olrpcp_filename):
        ds = xr.open_dataset(
            olrpcp_filename,
            # mask_and_scale=False,
            # decode_times=False,
        )
        precipitation = ds[pcp_varname].data.squeeze()
        cloudid_basetime = ds[time_dimname].data.squeeze()
        ds.close()

        # Convert OLR to Tb if olr2tb flag is set
        if olr2tb is not None:
            olr = ds[olr_varname].data.squeeze()
            tb = olr_to_tb(olr)
        else:
            tb = ds[tb_varname].data.squeeze()

        # Get dimensions of data
        ydim, xdim = np.shape(tb)

        # TODO: subset OLR/PCP data if domain size does not match landmask
        if (ny != ydim) | (nx != xdim):
            print(f'WARNING: OLR/PCP data domain size is NOT the same with landmask data!')
            print(f'Code will exit now.')
            sys.exit()

        # Number of clouds
        nmatchcloud = len(file_mcsnumber)

        if nmatchcloud > 0:
            # Define a list of 2D variables [tracks, times]
            var_names_2d = [
                "core_area",
                "cold_area",
                "ccs_area",
                "meanlat",
                "meanlon",
                "corecold_mintb",
                "corecold_meantb",
                "lat_mintb",
                "lon_mintb",
                "pf_npf",
                "pf_landfrac",
                "total_rain",
                "total_heavyrain",
                "rainrate_heavyrain",
            ]
            # CCS variables
            core_area = np.full(nmatchcloud, fillval_f, dtype=np.float32)
            cold_area = np.full(nmatchcloud, fillval_f, dtype=np.float32)
            ccs_area = np.full(nmatchcloud, fillval_f, dtype=np.float32)
            meanlat = np.full(nmatchcloud, fillval_f, dtype=np.float32)
            meanlon = np.full(nmatchcloud, fillval_f, dtype=np.float32)
            corecold_mintb = np.full(nmatchcloud, fillval_f, dtype=np.float32)
            corecold_meantb = np.full(nmatchcloud, fillval_f, dtype=np.float32)
            lat_mintb = np.full(nmatchcloud, fillval_f, dtype=np.float32)
            lon_mintb = np.full(nmatchcloud, fillval_f, dtype=np.float32)

            # PF variables 
            pf_npf = np.full(nmatchcloud, fillval, dtype=np.int16)
            pf_landfrac = np.full(nmatchcloud, fillval_f, dtype=np.float32)
            total_rain = np.full(nmatchcloud, fillval_f, dtype=np.float32)
            total_heavyrain = np.full(nmatchcloud, fillval_f, dtype=np.float32)
            rainrate_heavyrain = np.full(nmatchcloud, fillval_f, dtype=np.float32)
            pf_lon = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_lat = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_area = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_rainrate = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_skewness = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_maxrainrate = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_majoraxis = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_minoraxis = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_aspectratio = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_orientation = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_eccentricity = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_perimeter = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_lon_centroid = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_lat_centroid = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_lon_weightedcentroid = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_lat_weightedcentroid = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_accumrain = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_accumrainheavy = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_lon_maxrainrate = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            pf_lat_maxrainrate = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=np.float32)
            # basetime = np.full(nmatchcloud, fillval_f, dtype=float)

            # Loop over each matched cloud number
            for imatchcloud in range(nmatchcloud):

                ittcloudnumber = file_mcsnumber[imatchcloud]
                # ittmergecloudnumber = ir_mergecloudnumber[imatchcloud]
                # ittsplitcloudnumber = ir_splitcloudnumber[imatchcloud]
                # basetime[imatchcloud] = cloudid_basetime

                # Find MCS mask locations
                icloudlocationy, icloudlocationx = np.where(imcs_mask == ittcloudnumber)
                ncloudpix = len(icloudlocationy)
                # if ncloudpix > 0:
                # Get bounds
                miny, maxy = np.min(icloudlocationy), np.max(icloudlocationy)
                minx, maxx = np.min(icloudlocationx), np.max(icloudlocationx)

                # Isolate region over the cloud shield
                sub_mcs_mask = imcs_mask[miny:maxy, minx:maxx].copy()
                sub_tb_map = tb[miny:maxy, minx:maxx].copy()
                sub_rainrate_map = precipitation[miny:maxy, minx:maxx].copy()
                sub_lat = lat[miny:maxy, minx:maxx].copy()
                sub_lon = lon[miny:maxy, minx:maxx].copy()
                
                # Mask out areas outside of MCS
                sub_nonmcs = sub_mcs_mask != ittcloudnumber                
                sub_tb_map[sub_nonmcs] = np.NaN
                sub_rainrate_map[sub_nonmcs] = np.NaN
                sub_lat[sub_nonmcs] = np.NaN
                sub_lon[sub_nonmcs] = np.NaN

                # uniq_mcs_num = np.unique(sub_mcs_mask)
                # n_uniq_mcs = len(uniq_mcs_num)
                # if n_uniq_mcs > 2:
                #     import pdb; pdb.set_trace()

                # Get masks for cold core, cold cloud
                mask_core = sub_tb_map < cloudtb_core
                mask_cold = sub_tb_map < cloudtb_cold
                
                # Calculate cloud shield statistics
                core_npix = np.count_nonzero(mask_core)
                cold_npix = np.count_nonzero(mask_cold)
                if cold_npix > 0:
                    core_area[imatchcloud] = core_npix * pixel_radius ** 2
                    cold_area[imatchcloud] = cold_npix * pixel_radius ** 2
                    ccs_area[imatchcloud] = core_area[imatchcloud] + cold_area[imatchcloud]
                    meanlat[imatchcloud] = np.nanmean(sub_lat)
                    meanlon[imatchcloud] = np.nanmean(sub_lon)
                    corecold_mintb[imatchcloud] = np.nanmin(sub_tb_map)
                    corecold_meantb[imatchcloud] = np.nanmean(sub_tb_map)
                    # Get min Tb location
                    mintb_index = np.nanargmin(sub_tb_map)
                    # mintb_yid, mintb_xid = np.unravel_index(mintb_index, sub_tb_map.shape)
                    lat_mintb[imatchcloud] = sub_lat.flatten()[mintb_index]
                    lon_mintb[imatchcloud] = sub_lon.flatten()[mintb_index]


                # Calculate total rainfall within the cold cloud shield
                total_rain[imatchcloud] = np.nansum(sub_rainrate_map)
                idx_heavyrain = np.where(sub_rainrate_map > heavy_rainrate_thresh)
                if len(idx_heavyrain[0]) > 0:
                    total_heavyrain[imatchcloud] = np.nansum(
                        sub_rainrate_map[idx_heavyrain]
                    )
                    rainrate_heavyrain[imatchcloud] = np.nanmean(
                        sub_rainrate_map[idx_heavyrain]
                    )
                    
                ######################################################
                # Derive individual PF statistics
                # print("Calculating precipitation statistics")

                ipfy, ipfx = np.array(np.where(sub_rainrate_map > pf_rr_thres))
                nrainpix = len(ipfy)

                if nrainpix > 0:

                    # Calculate fraction of PF over land
                    if landmask is not None:
                        # Subset landmask to current cloud area
                        sublandmask = landmask[miny:maxy, minx:maxx]
                        # Count the number of grids within the specified land fraction range
                        npix_land = np.count_nonzero(
                            (sublandmask[ipfy, ipfx] >= np.min(landfrac_thresh)) & \
                            (sublandmask[ipfy, ipfx] <= np.max(landfrac_thresh))
                        )
                        if npix_land > 0:
                            pf_landfrac[imatchcloud] = \
                                float(npix_land) / float(nrainpix)
                        else:
                            pf_landfrac[imatchcloud] = 0
                    pass

                    ####################################################
                    ## Get dimensions of subsetted region
                    subdimy, subdimx = np.shape(sub_rainrate_map)

                    # Create binary map
                    binarypfmap = np.zeros((subdimy, subdimx), dtype=int)
                    binarypfmap[ipfy, ipfx] = 1

                    # Label precipitation features
                    pfnumberlabelmap, numpf = label(binarypfmap)

                    # Sort numpf then calculate stats
                    min_npix = np.ceil(pf_link_area_thresh / (pixel_radius ** 2)).astype(int)

                    # Sort and renumber PFs, and remove small PFs
                    pf_number, pf_npix = sort_renumber(pfnumberlabelmap, min_npix)
                    # Update number of PFs after sorting and renumbering
                    npf_new = np.nanmax(pf_number)
                    numpf = npf_new
                    pfnumberlabelmap = pf_number

                    if numpf > 0:
                        ###################################################
                        # print("PFs present, calculating statistics")

                        # Call function to calculate individual PF statistics
                        pf_stats_dict = calc_pf_stats(
                            fillval, fillval_f, heavy_rainrate_thresh,
                            lat, lon, minx, miny, nmaxpf, numpf,
                            pf_npix, pfnumberlabelmap, pixel_radius,
                            subdimx, subdimy, sub_rainrate_map,
                        )

                        # Save precipitation feature statisitcs
                        npf_save = pf_stats_dict["npf_save"]
                        pf_npf[imatchcloud] = np.copy(numpf)
                        pf_lon[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pflon"][0:npf_save]
                        pf_lat[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pflat"][0:npf_save]
                        pf_area[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pfnpix"][0:npf_save] * pixel_radius**2
                        pf_rainrate[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pfrainrate"][0:npf_save]
                        pf_maxrainrate[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pfmaxrainrate"][0:npf_save]
                        pf_skewness[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pfskewness"][0:npf_save]
                        pf_majoraxis[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pfmajoraxis"][0:npf_save]
                        pf_minoraxis[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pfminoraxis"][0:npf_save]
                        pf_aspectratio[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pfaspectratio"][0:npf_save]
                        pf_orientation[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pforientation"][0:npf_save]
                        pf_eccentricity[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pfeccentricity"][0:npf_save]
                        pf_lat_centroid[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pflat_centroid"][0:npf_save]
                        pf_lon_centroid[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pflon_centroid"][0:npf_save]
                        pf_lat_weightedcentroid[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pflat_weightedcentroid"][0:npf_save]
                        pf_lon_weightedcentroid[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pflon_weightedcentroid"][0:npf_save]
                        pf_accumrain[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pfaccumrain"][0:npf_save]
                        pf_accumrainheavy[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pfaccumrainheavy"][0:npf_save]
                        pf_perimeter[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pfperimeter"][0:npf_save]
                        pf_lon_maxrainrate[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pflon_maxrainrate"][0:npf_save]
                        pf_lat_maxrainrate[imatchcloud, 0:npf_save] = \
                            pf_stats_dict["pflat_maxrainrate"][0:npf_save]
                    # if numpf > 0:
                # if nrainpix > 0:
                # if ncloudpix > 0:
            # for imatchcloud in range(nmatchcloud):

            # Group outputs in dictionaries
            out_dict = {
                # CCS variables
                "core_area": core_area,
                "cold_area": cold_area,
                "ccs_area": ccs_area,
                "meanlat": meanlat,
                "meanlon": meanlon,
                "corecold_mintb": corecold_mintb,
                "corecold_meantb": corecold_meantb,
                "lat_mintb": lat_mintb,
                "lon_mintb": lon_mintb,
                # PF variables
                "pf_npf": pf_npf,
                "pf_lon": pf_lon,
                "pf_lat": pf_lat,
                "pf_area": pf_area,
                "pf_rainrate": pf_rainrate,
                "pf_skewness": pf_skewness,
                "pf_majoraxis": pf_majoraxis,
                "pf_minoraxis": pf_minoraxis,
                "pf_aspectratio": pf_aspectratio,
                "pf_orientation": pf_orientation,
                "pf_perimeter": pf_perimeter,
                "pf_eccentricity": pf_eccentricity,
                "pf_lon_centroid": pf_lon_centroid,
                "pf_lat_centroid": pf_lat_centroid,
                "pf_lon_weightedcentroid": pf_lon_weightedcentroid,
                "pf_lat_weightedcentroid": pf_lat_weightedcentroid,
                "pf_lon_maxrainrate": pf_lon_maxrainrate,
                "pf_lat_maxrainrate": pf_lat_maxrainrate,
                "pf_maxrainrate": pf_maxrainrate,
                "pf_accumrain": pf_accumrain,
                "pf_accumrainheavy": pf_accumrainheavy,
                "pf_landfrac": pf_landfrac,
                "total_rain": total_rain,
                "total_heavyrain": total_heavyrain,
                "rainrate_heavyrain": rainrate_heavyrain,
            }
            out_dict_attrs = {
                # CCS variables
                "core_area": {
                    "long_name": "Area of cold anvil",
                    "units": "km^2",
                    "_FillValue": fillval,
                },
                "cold_area": {
                    "long_name": "Area of cold core",
                    "units": "km^2",
                    "_FillValue": fillval,
                },
                "ccs_area":{
                    "long_name": "Area of cold cloud shield",
                    "units": "km^2",
                    "_FillValue": fillval,
                }, 
                "meanlat": {
                    "long_name": "Mean latitude of a feature",
                    "units": "degrees_north",
                    "_FillValue": fillval,
                }, 
                "meanlon": {
                    "long_name": "Mean longitude of a feature",
                    "units": "degrees_east",
                    "_FillValue": fillval,
                }, 
                "corecold_mintb": {
                    "long_name": "Minimum Tb in cold core + cold anvil area",
                    "units": "K",
                    "_FillValue": fillval,
                }, 
                "corecold_meantb": {
                    "long_name": "Mean Tb in cold core + cold anvil area",
                    "units": "K",
                    "_FillValue": fillval,
                }, 
                "lat_mintb": {
                    "long_name": "Latitude with min Tb",
                    "units": "degree",
                    "_FillValue": fillval,
                }, 
                "lon_mintb": {
                    "long_name": "Longitude with min Tb",
                    "units": "degree",
                    "_FillValue": fillval,
                }, 
                # PF variables
                "pf_npf": {
                    "long_name": "Number of PF in the cloud",
                    "units": "unitless",
                    "_FillValue": fillval,
                },
                "pf_lon": {
                    "long_name": "Mean longitude of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_lat": {
                    "long_name": "Mean latitude of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_area": {
                    "long_name": "Area of PF",
                    "units": "km^2",
                    "_FillValue": fillval_f,
                },
                "pf_rainrate": {
                    "long_name": "Mean rain rate of PF",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "pf_skewness": {
                    "long_name": "Rain rate skewness of PF",
                    "units": "unitless",
                    "_FillValue": fillval_f,
                },
                "pf_majoraxis": {
                    "long_name": "Major axis length of PF",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_minoraxis": {
                    "long_name": "Minor axis length of PF",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_aspectratio": {
                    "long_name": "Aspect ratio (major axis / minor axis) of PF",
                    "units": "unitless",
                    "_FillValue": fillval_f,
                },
                "pf_orientation": {
                    "long_name": "Orientation of major axis of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_eccentricity": {
                    "long_name": "Eccentricity of PF",
                    "units": "unitless",
                    "_FillValue": fillval_f,
                },
                "pf_perimeter": {
                    "long_name": "Perimeter of PF",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_lon_centroid": {
                    "long_name": "Centroid longitude of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_lat_centroid": {
                    "long_name": "Centroid latitude of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_lon_weightedcentroid": {
                    "long_name": "Weighted centroid longitude of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_lat_weightedcentroid": {
                    "long_name": "Weighted centroid latitude of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_maxrainrate": {
                    "long_name": "Max rain rate of PF",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "pf_accumrain": {
                    "long_name": "Accumulate precipitation of PF",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "pf_accumrainheavy": {
                    "long_name": "Accumulated heavy precipitation of PF",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                    "heavy_rainrate_threshold": heavy_rainrate_thresh,
                },
                "pf_landfrac": {
                    "long_name": "Fraction of PF over land",
                    "units": "fraction",
                    "_FillValue": fillval_f,
                },
                "total_rain": {
                    "long_name": "Total precipitation under cold cloud shield (all rainfall included)",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "total_heavyrain": {
                    "long_name": "Total heavy precipitation under cold cloud shield",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                    "heavy_rainrate_threshold": heavy_rainrate_thresh,
                },
                "rainrate_heavyrain": {
                    "long_name": "Mean heavy rain rate under cold cloud shield",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                    "heavy_rainrate_threshold": heavy_rainrate_thresh,
                },
                "pf_lon_maxrainrate": {
                    "long_name": "Longitude with max rain rate",
                    "units": "degree",
                    "_FillValue": fillval_f,
                },
                "pf_lat_maxrainrate": {
                    "long_name": "Latitude with max rain rate",
                    "units": "degree",
                    "_FillValue": fillval_f,
                },
            }
            # import pdb; pdb.set_trace()
            print(f'Done processing: {olrpcp_filename}')
            return out_dict, out_dict_attrs, var_names_2d

        else:
            print("No matching cloud found in: " + olrpcp_filename)

    else:
        print("cloudid file does not exist: " + olrpcp_filename)



if __name__ == "__main__":

    # Get inputs from command line
    config_file = sys.argv[1]
    tracker = sys.argv[2]

    print(f'Start time: {time.ctime(time.time())}')

    # Get runname and PHASE from config_file name
    parts = config_file.split("/")
    config_file_basename = parts[-1]
    # Config filname format: config_dyamond_PHASE_runname.yml
    runname = config_file_basename.split("_")[-1].split(".")[0]
    PHASE = config_file_basename.split("_")[-2].capitalize()
    print(f'{PHASE} {runname} {tracker}')

    # Input
    mask_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_mask/{PHASE}/{tracker}/'
    mask_file = f'{mask_dir}mcs_mask_{PHASE}_{runname}.nc'
    # Output
    out_dir = f'/pscratch/sd/f/feng045/DYAMOND/mcs_stats/{PHASE}/{tracker}/'
    out_filename = f'{out_dir}mcs_stats_{PHASE}_{runname}.nc'
    os.makedirs(out_dir, exist_ok=True)

    # DYAMOND phase start date
    if PHASE == 'Summer':
        reference_date = '2016-08-01'
    elif PHASE == 'Winter':
        reference_date = '2020-01-20'
    reference_time = np.datetime64(f'{reference_date}T00:00:00')

    # OLR/PCP data file suffix
    if runname == 'OBS':
        suffix = '_4km-pixel'
    else:
        suffix = ''

    # Parallel setup
    run_parallel = 1
    n_workers = 128

    # Number of times for the MCS track stats file
    ntimes_out = 600

    mask_varname = 'mcs_mask'
    # mask_varname = 'mcs_mask_no_mergesplit'

    # Check required input files
    if os.path.isfile(mask_file) == False:
        print(f'ERROR: mask file does not exist: {mask_file}')
        sys.exit(f'Code will exist now.')
    if os.path.isfile(config_file) == False:
        print(f'ERROR: config file does not exist: {config_file}')
        sys.exit(f'Code will exist now.')

    # Load configuration file
    stream = open(config_file, "r")
    config = yaml.full_load(stream)
    nmaxpf = config["nmaxpf"]
    data_dir = config["clouddata_path"]
    data_basename = config["databasename"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    pf_dimname = config["pf_dimname"]
    cloudtb_core = config['cloudtb_core']
    cloudtb_cold = config['cloudtb_cold']
    pf_rr_thres = config["pf_rr_thres"]
    pf_link_area_thresh = config["pf_link_area_thresh"]
    heavy_rainrate_thresh = config["heavy_rainrate_thresh"]
    pixel_radius = config["pixel_radius"]
    time_resolution_hour = config['datatimeresolution']
    fillval = config["fillval"]
    fillval_f = np.nan

    # Read MCS mask file
    print(f'Reading MCS mask file: {mask_file}')
    ds = xr.open_dataset(mask_file, mask_and_scale=False)
    ntimes = ds.sizes['time']
    # time_mcs_mask = ds['time']

    # Check time encoding from the mask file
    time_encoding = ds['time'].encoding.get('calendar', None)
    # Convert 'noleap' calendar time to datetime to DatetimeIndex (e.g., SCREAM)
    if time_encoding == 'noleap':
        time_DatetimeIndex = xr.cftime_range(start=ds['time'].values[0], periods=ntimes, freq='1H', calendar='noleap').to_datetimeindex()
        # Convert DatetimeIndex to DataArray, then replace the time coordinate in the DataSet
        time_mcs_mask = xr.DataArray(time_DatetimeIndex, coords={'time': time_DatetimeIndex}, dims='time')
        ds['time'] = time_mcs_mask
    else:
        time_mcs_mask = ds['time']

    # Get MCS mask and convert the type as int if needed
    if ds[mask_varname].dtype != int:
        mcs_mask = ds[mask_varname].astype(int)
    else:
        mcs_mask = ds[mask_varname]

    # Get max track number, this is the total number of tracks
    ntracks = mcs_mask.max().astype(int).item()
    print(f'Total number of times: {ntimes}')
    print(f'Total number of tracks: {ntracks}')

    # Find all unique track numbers at each time and save to a list
    # This should take ~1 min to run for 960 times
    print(f'Finding MCS track numbers at each time ...')
    # mcs_mask_np = mcs_mask.data
    list_mcs = []
    list_mcs_times = []
    for itime in range(0, ntimes):
        # Get unique MCS number at thie time
        # mcs_num = np.unique(mcs_mask_np[itime])
        mcs_num = np.unique(mcs_mask.isel(time=itime))

        # Find MCS number > 0 in the list (exclude 0 & NaN)
        # mcs_num_x0 = list(filter(lambda x: x != 0, mcs_num))
        mcs_num_x0 = list(filter(lambda x: x > 0, mcs_num))
        # Append MCS numbers to the list (if they exist)
        if len(mcs_num_x0) > 0:
            list_mcs.append(mcs_num_x0)
            list_mcs_times.append(time_mcs_mask.isel(time=itime).data)
        print(f'Time {itime}, number of MCS: {len(mcs_num_x0)}')
    
    # Convert list to array
    list_mcs_times = np.array(list_mcs_times)
    # Subset MCS mask to times when MCS exists
    mcs_mask = mcs_mask.sel(time=list_mcs_times)
    mcs_mask_np = mcs_mask.data

    # Create an empty datetime64[ns] array filled with NaT (Not a Time)
    dims = (ntracks, ntimes_out)
    base_times = np.full(dims, np.datetime64('NaT'), dtype='datetime64[ns]')
    # Create MCS time array with simple reference time
    mcs_times = np.full(dims, fillval, dtype=int)
    mcs_times_attr = {
        'long_name': 'time of MCS',
        'units': f'Hours since {reference_date} 00:00:00',
    }

    # Output track stats file coordinates
    coord_tracks = np.arange(0, ntracks)
    coord_times = np.arange(0, ntimes_out)
    coord_pfs = np.arange(0, nmaxpf)

    # Convert list of varying lengths to numpy array for faster operation
    list_mcs_np = np.fromiter((np.array(sublist) for sublist in list_mcs), dtype=object)
    # Find unique track numbers
    unique_mcs_tracknum = set(number for sublist in list_mcs for number in sublist)
    unique_mcs_tracknum = sorted(list(unique_mcs_tracknum))
    # import pdb; pdb.set_trace()

    print(f'Making base_times for each track ...')
    # Loop over each track
    # for tracknum in range(1, ntracks+1):
    for tracknum in unique_mcs_tracknum:
        ii = tracknum - 1  # track index
        # Find time indices for the track
        # tidx_tracknum = find_track_time_indices(list_mcs, tracknum)
        tidx_tracknum = np.where(np.array([tracknum in sublist for sublist in list_mcs_np]))[0]
        # import pdb; pdb.set_trace()
        # if tidx_tracknum:
        if len(tidx_tracknum) > 0:
            print(f'Track {ii}, duration: {len(tidx_tracknum)}')
            # base_times[ii, :len(tidx_tracknum)] = time_mcs_mask.isel(time=tidx_tracknum).data
            base_times[ii, :len(tidx_tracknum)] = list_mcs_times[tidx_tracknum]

            # Convert time values to hours since the reference time
            itime = list_mcs_times[tidx_tracknum]
            hours_since_reference = (itime - reference_time) / np.timedelta64(1, 'h')
            mcs_times[ii, :len(tidx_tracknum)] = hours_since_reference.astype(int).data
    # import pdb; pdb.set_trace()

    # Get track duration, start/end time
    mcs_exist = mcs_times > 0
    track_duration = np.nansum(mcs_exist, axis=1)
    # Track start time
    start_basetime = mcs_times[:,0]
    # Sum over time dimension for valid times, -1 to get the last valid time index for each track
    # This is the end time index of each track (i.e. +1 equals the duration of each track)
    end_time_idx = np.nansum(mcs_exist, axis=1)-1
    # Apply fancy indexing to mcs_times: a tuple that indicates for each track, get the end time index
    end_basetime = mcs_times[(np.arange(0,ntracks), end_time_idx)]
    track_duration_attr = {
        'long_name': 'Duration of each track',
        'units': 'unitless',
        'comments': 'Multiply by time_resolution_hour to convert to physical units'
    }    
    start_time_attr = {
        'long_name': 'Start time of each track',
        'units': f'Hours since {reference_date} 00:00:00',
    }
    end_time_attr = {
        'long_name': 'End time of each track',
        'units': f'Hours since {reference_date} 00:00:00',
    }

    # Remove tracks that do not exist (e.g., TOOCAN)
    track_idx_exist = np.where(track_duration > 0)[0]
    # Update dimension & coordinate
    ntracks = len(track_idx_exist)
    coord_tracks = coord_tracks[track_idx_exist]
    # Subset variables
    track_duration = track_duration[track_idx_exist]
    start_basetime = start_basetime[track_idx_exist]
    end_basetime = end_basetime[track_idx_exist]
    mcs_times = mcs_times[track_idx_exist, :]
    base_times = base_times[track_idx_exist, :]


    ################################################################
    # Define a dataset containing time-related variables
    varlist = {
        'track_duration': ([tracks_dimname], track_duration, track_duration_attr),
        'start_basetime': ([tracks_dimname], start_basetime, start_time_attr),
        'end_basetime': ([tracks_dimname], end_basetime, end_time_attr),
        'base_time': ([tracks_dimname, times_dimname], mcs_times, mcs_times_attr),
    }
    # Define coordinate list
    coordlist = {
        tracks_dimname: ([tracks_dimname], coord_tracks),
        times_dimname: ([times_dimname], coord_times),
    }
    # Define global attributes
    gattrlist = {}
    # Define Tb DataSet
    ds_tb = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # import pdb; pdb.set_trace()

    # Start Dask cluster
    if run_parallel == 1:
        # Set Dask temporary directory for workers
        dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        # Local cluster
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)


    #########################################################################################
    # Get unique base_times (files)
    unique_basetimes, ntracks_per_time = np.unique(base_times, return_counts=True)
    # Remove non-NaT values
    unique_basetimes = unique_basetimes[~pd.isnull(unique_basetimes)]
    # Number of unique times (i.e., pixel files)
    nfiles = len(unique_basetimes)

    # Create a list to store matchindices for each pixel file
    trackindices_all = []
    timeindices_all = []
    results = []

    print(f'Calculating statistics from pixel-files ...')
    # Loop over each pixel file to calculate PF statistics
    for ifile in range(nfiles):
    # for ifile in range(0, 2):
        # Convert base_time to string format
        ibt = unique_basetimes[ifile].astype('datetime64[ns]')
        idt = pd.to_datetime(ibt)
        idt_str = idt.strftime('%Y%m%d%H')
        # OLR/precipitation data filename
        olrpcp_filename = f"{data_dir}{data_basename}{idt_str}{suffix}.nc"

        # Find all matching time indices from base_times to the current time
        matchindices = np.array(np.where(base_times == ibt))
        # The returned match indices are for [tracks, times] dimensions respectively
        idx_track = matchindices[0]
        idx_time = matchindices[1]

        # Get MCS numbers for this time (file)
        file_mcsnumber = list_mcs[ifile]

        # Save matchindices for the current pixel file to the overall list
        trackindices_all.append(idx_track)
        timeindices_all.append(idx_time)

        # Call function to calculate PF stats
        imcs_mask = mcs_mask_np[ifile]
        # Serial
        if run_parallel == 0:
            result = featurestats_singlefile(
                olrpcp_filename,
                file_mcsnumber,
                imcs_mask,
                config,
            )
            # results.append(result)
        # Parallel
        elif run_parallel == 1:
            result = dask.delayed(featurestats_singlefile)(
                olrpcp_filename,
                file_mcsnumber,
                imcs_mask,
                config,
            )
        else:
            sys.exit('Valid parallelization flag not provided.')
        results.append(result)
    # for ifile in range(nfiles)

    if run_parallel == 0:
        final_result = results
    elif run_parallel == 1:
        # Trigger dask computation
        final_result = dask.compute(*results)
        wait(final_result)

        # Close the Dask cluster
        client.close()
        cluster.close()
    else:
        sys.exit('Valid parallelization flag not provided.')


    #########################################################################################
    # Create arrays to store output
    print("Collecting track PF statistics.")

    # maxtracklength = ntimes_out
    # numtracks = ntracks

    # Make a variable list and get attributes from one of the returned dictionaries
    # Loop over each return results till one that is not None
    counter = 0
    while counter < nfiles:
        if final_result[counter] is not None:
            var_names = list(final_result[counter][0].keys())
            # Get variable attributes
            var_attrs = final_result[counter][1]
            var_names_2d = final_result[counter][2]
            break
        counter += 1

    # Loop over variable list to create the dictionary entry
    pf_dict = {}
    pf_dict_attrs = {}
    for ivar in var_names:
        pf_dict[ivar] = np.full((ntracks, ntimes_out, nmaxpf), np.nan, dtype=np.float32)
        pf_dict_attrs[ivar] = var_attrs[ivar]
    for ivar in var_names_2d:
        pf_dict[ivar] = np.full((ntracks, ntimes_out), np.nan, dtype=np.float32)

    # Collect results
    for ifile in range(0, nfiles):
    # for ifile in range(0, 2):
        if final_result[ifile] is not None:
            # Get the return results for this pixel file
            # The result is a tuple: (out_dict, out_dict_attrs)
            # The first entry is the dictionary containing the variables
            iResult = final_result[ifile][0]

            # Get trackindices and timeindices for this file
            trackindices = trackindices_all[ifile]
            timeindices = timeindices_all[ifile]

            # Loop over each variable and assign values to output dictionary
            for ivar in var_names:
                if iResult[ivar].ndim == 1:
                    pf_dict[ivar][trackindices, timeindices] = iResult[ivar]
                if iResult[ivar].ndim == 2:
                    pf_dict[ivar][trackindices, timeindices, :] = iResult[ivar]

    # Define a dataset containing all PF variables
    varlist = {}
    # Define output variable dictionary
    for key, value in pf_dict.items():
        if value.ndim == 1:
            varlist[key] = ([tracks_dimname], value, pf_dict_attrs[key])
        if value.ndim == 2:
            varlist[key] = ([tracks_dimname, times_dimname], value, pf_dict_attrs[key])
        if value.ndim == 3:
            varlist[key] = ([tracks_dimname, times_dimname, pf_dimname], value, pf_dict_attrs[key])

    # Define coordinate list
    coordlist = {
        tracks_dimname: ([tracks_dimname], coord_tracks),
        times_dimname: ([times_dimname], coord_times),
        pf_dimname: ([pf_dimname], coord_pfs),
    }

    # Define global attributes
    gattrlist = {
        "title": f'{PHASE} {runname} MCS statistics file',
        "tracker": tracker,
        "pixel_radius_km": pixel_radius,
        "time_resolution_hour": time_resolution_hour,
        "tb_core_thresh": cloudtb_core,
        "tb_cold_thresh": cloudtb_cold,
        "nmaxpf": nmaxpf,
        "PF_rainrate_thresh": config["pf_rr_thres"],
        "heavy_rainrate_thresh": config["heavy_rainrate_thresh"],
        "landfrac_thresh": config["landfrac_thresh"],
    }

    # Define output Xarray dataset
    ds_pf = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)
    # Merge Tb and PF datasets
    dsout = xr.merge([ds_tb, ds_pf], compat="override", combine_attrs="no_conflicts")
    # Update time stamp
    dsout.attrs["Created_on"] = time.ctime(time.time())

    #########################################################################################
    # Save output to netCDF file
    print("Saving data ...")
    print((time.ctime()))

    # Delete file if it already exists
    if os.path.isfile(out_filename):
        os.remove(out_filename)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    dsout.to_netcdf(path=out_filename, mode="w",
                    format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    print(f"{out_filename}")
    print(f'End time: {time.ctime(time.time())}')