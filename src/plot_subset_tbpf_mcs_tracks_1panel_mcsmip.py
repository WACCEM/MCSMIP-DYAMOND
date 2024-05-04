"""
Ploting MCS tracks on Tb, precipitation snapshots for a subset domain.

>python plot_subset_tbpf_mcs_tracks_mcsmip.py -s STARTDATE -e ENDDATE  
Optional arguments:
-p 0 (serial), 1 (parallel)
--n_workers Number of workers for parallel
--extent lonmin lonmax latmin latmax (subset domain boundary)
--subset 0 (no), 1 (yes) (subset data before plotting)
--figsize width height (figure size in inches)
--output output_directory (output figure directory)
--figbasename figure base name (output figure base name)
--trackstats_file MCS track stats file name (optional, if different from robust MCS track stats file)
--pixel_path Pixel-level tracknumber mask files directory (optional, if different from robust MCS pixel files)
--phase DYAMOND phase (Summer or Winter)
--runname DYAMOND data runname (OBS or model name)
--tracker MCS tracker name

Zhe Feng, PNNL
contact: Zhe.Feng@pnnl.gov
"""

import argparse
import numpy as np
import os, sys
import xarray as xr
import pandas as pd
from scipy.ndimage import binary_erosion
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import colorcet as cc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# For non-gui matplotlib back end
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
mpl.use('agg')
import dask
from dask.distributed import Client, LocalCluster
import warnings
warnings.filterwarnings("ignore")
# from pyflextrkr.ft_utilities import load_config, subset_files_timerange

#-----------------------------------------------------------------------
def parse_cmd_args():
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Plot MCS tracks on Tb & precipitation snapshots for a user-defined subset domain."
    )
    parser.add_argument("-s", "--start", help="first time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-e", "--end", help="last time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-c", "--config", help="yaml config file for tracking", required=False)
    parser.add_argument("-p", "--parallel", help="flag to run in parallel (0:serial, 1:parallel)", type=int, default=0)
    parser.add_argument("--n_workers", help="Number of workers for parallel", type=int, default=1)
    parser.add_argument("--extent", nargs='+', help="map extent (lonmin, lonmax, latmin, latmax)", type=float, default=None)
    parser.add_argument("--subset", help="flag to subset data (0:no, 1:yes)", type=int, default=0)
    parser.add_argument("--figsize", nargs='+', help="figure size (width, height) in inches", type=float, default=None)
    parser.add_argument("--output", help="ouput directory", default=None)
    parser.add_argument("--figbasename", help="output figure base name", default="")
    parser.add_argument("--trackstats_file", help="MCS track stats file name", default=None)
    parser.add_argument("--pixel_path", help="Pixel-level tracknumer mask files directory", default=None)
    parser.add_argument("--phase", help="DYAMOND phase (Summer or Winter)", required=True)
    parser.add_argument("--runname", help="DYAMOND data runname (OBS or model name)", required=True)
    parser.add_argument("--tracker", help="MCS tracker name", required=True)
    args = parser.parse_args()

    # Put arguments in a dictionary
    args_dict = {
        'start_datetime': args.start,
        'end_datetime': args.end,
        'run_parallel': args.parallel,
        'n_workers': args.n_workers,
        'config_file': args.config,
        'extent': args.extent,
        'subset': args.subset,
        'figsize': args.figsize,
        'out_dir': args.output,
        'figbasename': args.figbasename,
        'trackstats_file': args.trackstats_file,
        'pixeltracking_path': args.pixel_path,
        'phase': args.phase,
        'runname': args.runname,
        'tracker': args.tracker,
    }

    return args_dict

#--------------------------------------------------------------------------
def olr_to_tb(OLR):
    """
    Convert OLR to IR brightness temperature.

    Args:
        OLR: np.array
            Outgoing longwave radiation
    
    Returns:
        tb: np.array
            Brightness temperature
    """
    # Calculate brightness temperature
    # (1984) as given in Yang and Slingo (2001)
    # Tf = tb(a+b*Tb) where a = 1.228 and b = -1.106e-3 K^-1
    # OLR = sigma*Tf^4 
    # where sigma = Stefan-Boltzmann constant = 5.67x10^-8 W m^-2 K^-4
    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8 # W m^-2 K^-4
    tf = (OLR/sigma)**0.25
    tb = (-a + np.sqrt(a**2 + 4*b*tf))/(2*b)
    return tb

#--------------------------------------------------------------------------
def make_dilation_structure(dilate_radius, dx, dy):
    """
    Make a circular dilation structure

    Args:
        dilate_radius: float
            Dilation radius [kilometer].
        dx: float
            Grid spacing in x-direction [kilometer].
        dy: float
            Grid spacing in y-direction [kilometer]. 
    
    Returns:
        struc: np.array
            Dilation structure array.
    """
    # Convert radius to number grids
    rad_gridx = int(dilate_radius / dx)
    rad_gridy = int(dilate_radius / dy)
    xgrd, ygrd = np.ogrid[-rad_gridx:rad_gridx+1, -rad_gridy:rad_gridy+1]
    # Make dilation structure
    strc = xgrd*xgrd + ygrd*ygrd <= (dilate_radius / dx) * (dilate_radius / dy)
    return strc

#-----------------------------------------------------------------------
def label_perimeter(tracknumber, dilationstructure):
    """
    Labels the perimeter on a 2D map from object tracknumber masks.
    """
    # Get unique tracknumbers that is no nan
    tracknumber_unique = np.unique(tracknumber[~np.isnan(tracknumber)]).astype(np.int32)

    # Make an array to store the perimeter
    tracknumber_perim = np.zeros(tracknumber.shape, dtype=np.int32)

    # Loop over each tracknumbers
    for ii in tracknumber_unique:
        # Isolate the track mask
        itn = tracknumber == ii
        # Erode the mask by 1 pixel
        itn_erode = binary_erosion(itn, structure=dilationstructure).astype(itn.dtype)
        # Subtract the eroded area to get the perimeter
        iperim = np.logical_xor(itn, itn_erode)
        # Label the perimeter pixels with the track number
        tracknumber_perim[iperim == 1] = ii

    return tracknumber_perim

#-----------------------------------------------------------------------
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """ 
    Truncate colormap.
    """
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#-----------------------------------------------------------------------
def get_track_stats(trackstats_file, start_datetime, end_datetime, dt_thres, map_extent=None, buffer=1.0):
    """
    Subset tracks statistics data within start/end datetime

    Args:
        trackstats_file: string
            Track statistics file name.
        start_datetime: string
            Start datetime to subset tracks.
        end_datetime: dstring
            End datetime to subset tracks.
        dt_thres: timedelta
            A timedelta threshold to retain tracks.
        map_extent: list, optional, default=None
            A list with boundary to subset the tracks ([lonmin, lonmax, latmin, latmax])
        buffer: float, optional, default=1.0
            A buffer value for subset boundary.
            
    Returns:
        track_dict: dictionary
            Dictionary containing track stats data.
    """
    # Read track stats file
    dss = xr.open_dataset(trackstats_file)
    stats_starttime = dss['base_time'].isel(times=0)
    stats_startlon = dss['meanlon'].isel(times=0)
    stats_startlat = dss['meanlat'].isel(times=0)
    # Convert input datetime to np.datetime64
    stime = np.datetime64(start_datetime)
    etime = np.datetime64(end_datetime)
    time_res = dss.attrs['time_resolution_hour']

    # Find tracks initiated within the time window
    if map_extent is None:
        idx = np.where((stats_starttime >= stime) & (stats_starttime <= etime))[0]
    else:
    # If map_extent is provided, also subset the tracks that start within the domain
        # Get subset domain boundary
        lonmin, lonmax = map_extent[0]-buffer, map_extent[1]+buffer
        latmin, latmax = map_extent[2]-buffer, map_extent[3]+buffer
        idx = np.where(
            (stats_starttime >= stime) & (stats_starttime <= etime) & 
            (stats_startlon >= lonmin) & (stats_startlon <= lonmax) &
            (stats_startlat >= latmin) & (stats_startlat <= latmax)
        )[0]
    ntracks = len(idx)
    print(f'Number of tracks within input period: {ntracks}')

    # Subset these tracks and put in a dictionary    
    track_dict = {
        'ntracks': ntracks,
        'lifetime': dss['track_duration'].isel(tracks=idx) * time_res,
        'track_bt': dss['base_time'].isel(tracks=idx),
        'track_ccs_lon': dss['meanlon'].isel(tracks=idx),
        'track_ccs_lat': dss['meanlat'].isel(tracks=idx),
        'track_pf_lon': dss['pf_lon_centroid'].isel(tracks=idx, nmaxpf=0),
        'track_pf_lat': dss['pf_lat_centroid'].isel(tracks=idx, nmaxpf=0),
        'track_pf_diam': 2 * np.sqrt(dss['pf_area'].isel(tracks=idx, nmaxpf=0) / np.pi),
        'dt_thres': dt_thres,
        'time_res': time_res,
    }
    
    return track_dict

#-----------------------------------------------------------------------
def plot_map_2panels(pixel_dict, plot_info, map_info, track_dict):
    """
    Plot 2 rows with Tb, Precipitation and MCS tracks snapshot.

    Args:
        pixel_dict: dictionary
            Dictionary containing pixel data variables.
        plot_info: dictionary
            Dictionary containing plotting setup variables.
        map_info: dictionary
            Dictionary containing map boundary info.
        track_dict: dictionary
            Dictionary containing tracking data variables.
            
    Returns:
        fig: object
            Figure handle.
    """
        
    # Get pixel data from dictionary
    lon = pixel_dict['lon']
    lat = pixel_dict['lat']
    tb = pixel_dict['tb']
    pcp = pixel_dict['pcp']
    tracknumber = pixel_dict['tracknumber']
    tn_perim = pixel_dict['tracknumber_perim']
    # pixel_bt = pixel_dict['pixel_bt']
    pixel_datetime64 = pixel_dict['pixel_datetime64']
    # Get track data from dictionary
    ntracks = track_dict['ntracks']
    lifetime = track_dict['lifetime']
    track_bt = track_dict['track_bt']
    track_ccs_lon = track_dict['track_ccs_lon']
    track_ccs_lat = track_dict['track_ccs_lat']
    track_pf_lon = track_dict['track_pf_lon']
    track_pf_lat = track_dict['track_pf_lat']
    track_pf_diam = track_dict['track_pf_diam']
    dt_thres = track_dict['dt_thres']
    time_res = track_dict['time_res']
    # Get plot info from dictionary
    plot_pcp = plot_info['plot_pcp']
    levels = plot_info['levels']
    cmaps = plot_info['cmaps']
    tb_alpha = plot_info['tb_alpha']
    pcp_alpha = plot_info['pcp_alpha']
    titles = plot_info['titles'] 
    cblabels = plot_info['cblabels']
    cbticks = plot_info['cbticks']
    fontsize = plot_info['fontsize']
    marker_size = plot_info['marker_size']
    tracknumber_fontsize = plot_info['tracknumber_fontsize']
    trackpath_linewidth = plot_info['trackpath_linewidth']
    pfdiam_linewidth = plot_info['pfdiam_linewidth']
    trackpath_color = plot_info['trackpath_color']
    mcsperim_color = plot_info['mcsperim_color']
    pfdiam_color = plot_info['pfdiam_color']
    pfdiam_scale = plot_info['pfdiam_scale']
    map_edgecolor = plot_info['map_edgecolor']
    map_resolution = plot_info['map_resolution']
    timestr = plot_info['timestr']
    figname = plot_info['figname']
    figsize = plot_info['figsize']
    dpi = plot_info['dpi']
    # Map domain, lat/lon ticks, background map features
    map_extent = map_info['map_extent']
    lonv = map_info.get('lonv', None)
    latv = map_info.get('latv', None)
    draw_border = map_info.get('draw_border', False)
    draw_state = map_info.get('draw_state', False)
    central_longitude = map_info.get('central_longitude', None)

    # Time difference matching pixel-time and track time
    dt_match = 1  # [min]
    
    # Marker style for tracks
    marker_style = dict(edgecolor=trackpath_color, facecolor=trackpath_color, linestyle='-', marker='o')

    # Set up map projection
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    data_proj = ccrs.PlateCarree(central_longitude=0)
    land = cfeature.NaturalEarthFeature('physical', 'land', map_resolution)
    borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', map_resolution)
    states = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lakes', map_resolution)

    # Set up figure
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['font.family'] = 'Helvetica'
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='w')

    # Set GridSpec for left (plot) and right (colorbars)
    gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1, 0.1])
    gs.update(wspace=0.05, left=0.05, right=0.95, top=0.92, bottom=0.08)
    # Use GridSpecFromSubplotSpec for panel and colorbar
    gs_cb = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], height_ratios=[1], width_ratios=[0.01,0.01], wspace=5)
    # ax1 = plt.subplot(gs[0], projection=proj)
    cax1 = plt.subplot(gs_cb[0])
    if plot_pcp == True:
        cax2 = plt.subplot(gs_cb[1])
    # Figure title: time
    fig.text(0.5, 0.96, timestr, fontsize=fontsize*1.4, ha='center')

    #################################################################
    # Tb Panel
    ax1 = plt.subplot(gs[0,0], projection=proj)
    ax1.set_extent(map_extent, crs=data_proj)
    ax1.add_feature(land, facecolor='none', edgecolor=map_edgecolor, zorder=4)
    if draw_border == True:
        ax1.add_feature(borders, edgecolor=map_edgecolor, facecolor='none', linewidth=0.8, zorder=4)
    if draw_state == True:
        ax1.add_feature(states, edgecolor=map_edgecolor, facecolor='none', linewidth=0.8, zorder=4)
    ax1.set_aspect('auto', adjustable=None)
    ax1.set_title(titles, loc='left')
    gl = ax1.gridlines(crs=data_proj, draw_labels=True, linestyle='--', linewidth=0.5)
    gl.right_labels = False
    gl.top_labels = False
    if (lonv is not None) & (latv is not None):
        gl.xlocator = mpl.ticker.FixedLocator(lonv)
        gl.ylocator = mpl.ticker.FixedLocator(latv)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()        
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)

    # Tb
    cmap = plt.get_cmap(cmaps['tb_cmap'])
    norm = mpl.colors.BoundaryNorm(levels['tb_levels'], ncolors=cmap.N, clip=True)
    tb_masked = np.ma.masked_where((np.isnan(tb)), tb)
    cf1 = ax1.pcolormesh(lon, lat, tb_masked, norm=norm, cmap=cmap, transform=data_proj, zorder=2, alpha=tb_alpha)
    # Overplot MCS boundary
    cmap = plt.get_cmap(cmaps['tn_cmap'])
    Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
    tn1 = ax1.pcolormesh(lon, lat, Tn, vmin=min(levels['tn_levels']), vmax=max(levels['tn_levels']),
                         cmap=cmap, transform=data_proj, zorder=4, alpha=1)
    # Precipitation
    if plot_pcp == True:
        cmap = plt.get_cmap(cmaps['pcp_cmap'])
        norm = mpl.colors.BoundaryNorm(levels['pcp_levels'], ncolors=cmap.N, clip=True)
        pcp_masked = np.ma.masked_where(((pcp < min(levels['pcp_levels']))), pcp)
        cf2 = ax1.pcolormesh(lon, lat, pcp_masked, norm=norm, cmap=cmap, transform=data_proj, zorder=2, alpha=pcp_alpha)
    # Tb Colorbar
    cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels['tb_label'], ticks=cbticks['tb_ticks'],
                       extend='both', orientation='vertical')
    # Precipitation Colorbar
    if plot_pcp == True:
        cb2 = plt.colorbar(cf2, cax=cax2, label=cblabels['pcp_label'], ticks=cbticks['pcp_ticks'],
                        extend='both', orientation='vertical')

    #################################################################
    # Plot track centroids and paths
    for itrack in range(0, ntracks):
        # Get duration of the track
        ilifetime = lifetime.data[itrack]
        itracknum = lifetime.tracks.data[itrack]+1
        idur = (ilifetime / time_res).astype(int)
        idiam = track_pf_diam.data[itrack,:idur]
        # Get basetime of the track and the track end time, round down the time to the nearest hour
        ibt = track_bt.dt.floor('h').data[itrack,:idur]
        ibt_end = np.nanmax(ibt)
        
        # Compute time difference between current pixel-level data time and the end time of the track
        idt_end = (pixel_datetime64 - ibt_end).astype('timedelta64[h]')

        # Proceed if time difference is <= threshold
        # This means for tracks that end longer than the time threshold are not plotted
        if (idt_end <= dt_thres):
            # Find times in track data <= current pixel-level file time
            idx_cut = np.where(ibt <= pixel_datetime64)[0]
            idur_cut = len(idx_cut)
            if (idur_cut > 0):
                # Track path
                color_vals = np.repeat(ilifetime, idur_cut)
                size_vals = np.repeat(marker_size, idur_cut)
                size_vals[0] = marker_size * 2
                cc1 = ax1.plot(track_ccs_lon.data[itrack,idx_cut], track_ccs_lat.data[itrack,idx_cut],
                               lw=trackpath_linewidth, ls='-', color=trackpath_color, transform=data_proj, zorder=5)
                # Initiation location
                cl1 = ax1.scatter(track_ccs_lon.data[itrack,0], track_ccs_lat.data[itrack,0], s=marker_size*2,
                                  transform=data_proj, zorder=5, **marker_style)
                
        # Find the closest time from track times
        idt = np.abs((ibt - pixel_datetime64).astype('timedelta64[m]'))
        # idt = np.abs((ibt - pixel_bt).astype('timedelta64[m]'))
        idx_match = np.argmin(idt)
        idt_match = idt[idx_match]
        # Get CCS center lat/lon from the matched tracks
        _iccslon = track_ccs_lon.data[itrack,idx_match]
        _iccslat = track_ccs_lat.data[itrack,idx_match]
        # Get PF radius from the matched tracks
        _irad = idiam[idx_match] / 2
        _ilon = track_pf_lon.data[itrack,idx_match]
        _ilat = track_pf_lat.data[itrack,idx_match]
        # Proceed if time difference is < dt_match
        dt_match64 = np.timedelta64(dt_match, 'm')
        if (idt_match < dt_match64):
            # # Plot PF diameter circle
            # if ~np.isnan(_irad):
            #     ipfcircle = ax1.tissot(rad_km=_irad*2, lons=_ilon, lats=_ilat, n_samples=100,
            #                            facecolor='None', edgecolor=pfdiam_color, lw=pfdiam_linewidth, zorder=3)
            # Overplot tracknumbers at current frame
            if (_iccslon > map_extent[0]) & (_iccslon < map_extent[1]) & \
                    (_iccslat > map_extent[2]) & (_iccslat < map_extent[3]):
                ax1.text(_iccslon+0.05, _iccslat+0.05, f'{itracknum:.0f}', color='k', size=tracknumber_fontsize,
                         ha='left', va='center', weight='bold', transform=data_proj, zorder=6)
    
    # Custom legend for track paths
    legend_elements1 = [
        mpl.lines.Line2D([0], [0], color=trackpath_color, marker='o', lw=trackpath_linewidth, label='MCS Tracks'),
        # mpl.lines.Line2D([0], [0], marker='o', lw=0, markerfacecolor='None',
        #                  markeredgecolor=mcsperim_color, markersize=12, label='MCS Mask'),
    ]
    ax1.legend(handles=legend_elements1, loc='lower right')
    
    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    canvas.print_png(figname)
    fig.savefig(figname)
    
    return fig

#-----------------------------------------------------------------------
def work_for_time_loop(olr_file, mask_file, track_dict, map_info, plot_info):
    """
    Work with a pixel-level file.

    Args:
        olr_file: string
            OLR & precipitation file name.
        mask_file: string
            MCS mask file name.
        track_dict: dictionary
            Dictionary containing tracking data variables.
        map_info: dictionary
            Dictionary containing map boundary info.
        plot_info: dictionary
            Dictionary containing plotting info.
            
    Returns:
        1: success.            
    """

    map_extent = map_info.get('map_extent', None)
    buffer = map_info.get('buffer', 0)
    perim_thick = plot_info.get('perim_thick')

    # Read OLR/precipitation file, subset to input datetime range
    dso = xr.open_dataset(olr_file).sel(time=slice(start_datetime, end_datetime))

    # Read MCS mask file, subset to intput datetime range
    dsm = xr.open_dataset(mask_file).sel(time=slice(start_datetime, end_datetime))

    nt_dso = dso.sizes['time']
    nt_dsm = dsm.sizes['time']
    # Replace the lat/lon coordinates
    if nt_dso == nt_dsm:
        dsm = dsm.assign_coords({'time':dso['time'], 'lon':dso['lon'], 'lat':dso['lat']})
    else:
        print(f'ERROR: number of subset times not the same between OLR/PCP and MCS mask files!')
        print(f'Code will exit now.')
        sys.exit()

    if 'OBS' not in runname:
        # Convert OLR to Tb
        tb = olr_to_tb(dso['olr'])
        # Add Tb to DataSet
        dso['Tb'] = tb
    
    # Combine OLR/precipitation and MCS mask DataSets
    ds = xr.combine_by_coords([dsm, dso], combine_attrs='drop_conflicts')
    # lon2d, lat2d = np.meshgrid(ds['lon'], ds['lat'])
    # import pdb; pdb.set_trace()

    # Get map extent from data
    if map_extent is None:
        lonmin = dsm['longitude'].min().item()
        lonmax = dsm['longitude'].max().item()
        latmin = dsm['latitude'].min().item()
        latmax = dsm['latitude'].max().item()
        map_extent = [lonmin, lonmax, latmin, latmax]
        map_info['map_extent'] = map_extent
        map_info['subset'] = subset

    # Make dilation structure (larger values make thicker outlines)
    # perim_thick = 6
    # dilationstructure = np.zeros((perim_thick+1,perim_thick+1), dtype=int)
    # dilationstructure[1:perim_thick, 1:perim_thick] = 1

    # Make a dilation structure
    dilationstructure = make_dilation_structure(perim_thick, pixel_radius, pixel_radius)
    plot_info['dilationstructure'] = dilationstructure

    # Get tracknumbers
    tn = ds['mcs_mask'].squeeze()
    # Only plot if there is track in the frame
    if (np.nanmax(tn) > 0):

        # Tracknumber color levels for MCS masks (limit to 256 to fit in a colormap)
        tracknumbers = track_dict['lifetime'].tracks
        tn_nlev = np.min([len(tracknumbers), 256])
        tn_levels = np.linspace(np.min(tracknumbers)+1, np.max(tracknumbers)+1, tn_nlev)
        # Add to plot_info dictionary
        plot_info['levels']['tn_levels'] = tn_levels

        # Subset pixel data within the map domain        
        if subset == 1:            
            lonmin, lonmax = map_extent[0]-buffer, map_extent[1]+buffer
            latmin, latmax = map_extent[2]-buffer, map_extent[3]+buffer
            mask = (ds['longitude'] >= lonmin) & (ds['longitude'] <= lonmax) & \
                   (ds['latitude'] >= latmin) & (ds['latitude'] <= latmax)
            ds_sub = ds.where(mask == True, drop=True)
            # ds_sub = ds.sel(lon=slice(lonmin, lonmax), lat=slice(latmin, latmax))
        else:
            ds_sub = ds

        # Get number of times in the DataSet
        ntimes = ds_sub.sizes['time']

        # Serial
        if run_parallel == 0:
            for itime in range(ntimes):
                ds_t = ds_sub.isel(time=itime)
                # print(datafiles[ifile])
                result = call_plot_func(ds_t, track_dict, map_info, plot_info,)

        # Parallel option
        elif run_parallel == 1:
            # Set Dask temporary directory for workers
            dask.config.set({'temporary-directory': dask_tmp_dir})
            # Initialize dask
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
            client = Client(cluster)
            results = []
            for itime in range(ntimes):
                ds_t = ds_sub.isel(time=itime)
                result = dask.delayed(call_plot_func)(
                    ds_t, track_dict, map_info, plot_info,
                )
                results.append(result)

            # Trigger dask computation
            final_result = dask.compute(*results)
            # Close Dask cluster
            cluster.close()
            client.close()

    ds.close()
    return 1

#-----------------------------------------------------------------------
def call_plot_func(ds, track_dict, map_info, plot_info):

    pixel_time = ds['time']
    tracknumber_sub = ds['mcs_mask']
    dilationstructure = plot_info.get('dilationstructure')

    # Check the calendar type
    calendar_type = pixel_time.encoding.get('calendar', None)
    if calendar_type == '365_day':
        # # Convert to standard Python datetime object (to deal with cftime.DatetimeNoLeap, e.g., SCREAM)
        # pixel_datetime = datetime.datetime(*pixel_time.item().timetuple()[:6])
        # # Convert to datetime64
        # pixel_datetime64 = np.datetime64(pixel_datetime)
        # Convert to datetime64
        pixel_datetime64 = np.datetime64(pixel_time.item(), 's')
    else:
        pixel_datetime64 = pd.to_datetime(pixel_time.item()).to_datetime64()
    
    # Get object perimeters
    tn_perim = label_perimeter(tracknumber_sub.data, dilationstructure)
    
    # Plotting variables
    fdatetime = ds['time'].dt.strftime('%Y%m%d_%H%M%S').item()
    timestr = ds['time'].dt.strftime('%Y-%m-%d %H:%M:%S UTC').item()
    # fdatetime = ds['time'].data.item().strftime('%Y%m%d_%H%M%S')
    # timestr = ds['time'].data.item().strftime('%Y-%m-%d %H:%M:%S UTC')
    # fdatetime = pd.to_datetime(ds['time'].data.item()).strftime('%Y%m%d_%H%M%S')
    # timestr = pd.to_datetime(ds['time'].data.item()).strftime('%Y-%m-%d %H:%M:%S UTC')
    # timestr = f'{timestr} ({runname}) [{tracker}]'
    figname = f'{figdir}{figbasename}{fdatetime}.png'

    # Put pixel data in a dictionary
    pixel_dict = {
        'lon': ds['longitude'],
        'lat': ds['latitude'],
        'tb': ds['Tb'],
        'pcp': ds['precipitation'],
        'tracknumber': tracknumber_sub,
        'tracknumber_perim': tn_perim,
        'pixel_datetime64': pixel_datetime64,
        # 'pixel_bt': pixel_bt,
    }
    plot_info['timestr'] = timestr
    plot_info['figname'] = figname
    # import pdb; pdb.set_trace()

    # Call plotting function
    fig = plot_map_2panels(pixel_dict, plot_info, map_info, track_dict)
    plt.close(fig)
    print(figname)

    return figname



if __name__ == "__main__":

    # Get the command-line arguments...
    args_dict = parse_cmd_args()
    start_datetime = args_dict.get('start_datetime')
    end_datetime = args_dict.get('end_datetime')
    run_parallel = args_dict.get('run_parallel')
    n_workers = args_dict.get('n_workers')
    # config_file = args_dict.get('config_file')
    # panel_orientation = args_dict.get('panel_orientation')
    map_extent = args_dict.get('extent')
    subset = args_dict.get('subset')
    figsize = args_dict.get('figsize')
    out_dir = args_dict.get('out_dir')
    figbasename = args_dict.get('figbasename')
    trackstats_file = args_dict.get('trackstats_file')
    pixeltracking_path = args_dict.get('pixeltracking_path')
    phase = args_dict.get('phase')
    runname = args_dict.get('runname')
    tracker = args_dict.get('tracker')

    dask_tmp_dir = "/tmp"


    # Determine the figsize based on lat/lon ratio
    if (figsize is None):
        # If map_extent is specified, calculate aspect ratio
        if (map_extent is not None):
            # Get map aspect ratio from map_extent (minlon, maxlon, minlat, maxlat)
            lon_span = map_extent[1] - map_extent[0]
            lat_span = map_extent[3] - map_extent[2]
            fig_ratio_yx = lat_span / lon_span

            figsize_x = 12
            figsize_y = figsize_x * fig_ratio_yx
            figsize_y = float("{:.2f}".format(figsize_y))  # round to 2 decimal digits
            figsize = [figsize_x, figsize_y]
        else:
            figsize = [10, 10]

    # Specify plotting info
    # Precipitation color levels
    pcp_levels = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
    pcp_ticks = pcp_levels
    # Tb color levels
    tb_levels = np.arange(180, 300.1, 1)
    tb_ticks = np.arange(180, 300.1, 20)
    levels = {'tb_levels': tb_levels, 'pcp_levels': pcp_levels}
    # Colorbar ticks & labels
    cbticks = {'tb_ticks': tb_ticks, 'pcp_ticks': pcp_ticks}
    cblabels = {'tb_label': 'Tb (K)', 'pcp_label': 'Precipitation (mm h$^{-1}$)'}
    # Colormaps
    # tb_cmap = 'Greys'
    tb_cmap = cc.cm['gray_r']
    pcp_cmap = cc.cm["CET_R1"]
    tn_cmap = cc.cm["glasbey_dark"]
    # tb_cmap = truncate_colormap(plt.get_cmap(tb_cmap), minval=0.01, maxval=0.99)
    # pcp_cmap = truncate_colormap(plt.get_cmap(pcp_cmap), minval=0, maxval=1.0)
    # tn_cmap = cc.cm["glasbey_light"]    
    # tn_cmap = cc.cm["glasbey"]
    cmaps = {'tb_cmap': tb_cmap, 'pcp_cmap': pcp_cmap, 'tn_cmap': tn_cmap}
    # titles = {'tb_title': 'IR Brightness Temperature, Precipitation, Tracked MCS (Outline)'}
    titles = f'{runname} [{tracker}]'
    plot_info = {
        'plot_pcp': True,
        'levels': levels,
        'cmaps': cmaps,
        'titles': titles,
        'cbticks': cbticks,
        'cblabels': cblabels,
        'tb_alpha': 0.9,
        'pcp_alpha': 0.3,
        'fontsize': 10,
        'dpi': 200,
        'marker_size': 10,
        'trackpath_linewidth': 1.5,
        'tracknumber_fontsize': 12,
        # 'trackpath_color': 'purple',
        'trackpath_color': 'blueviolet',
        'perim_thick': 10,  # [km]
        'mcsperim_color': 'magenta',
        'pfdiam_linewidth': 1,
        'pfdiam_scale': 1,  # scaling ratio for PF diameter circle (1: no scaling)
        'pfdiam_color': 'magenta',
        'map_edgecolor': 'k',
        'map_resolution': '50m',
        'map_central_lon': 180,
        'figsize': figsize,
        'figbasename': figbasename,
    }

    # Customize lat/lon labels
    lonv = None
    latv = None
    # Map projection central_longitude
    if (map_extent[0] <= 0) & (map_extent[1] > 0):
        central_longitude = 0
    else:
        central_longitude = 180
    # Put map info in a dictionary
    map_info = {
        'map_extent': map_extent,
        'subset': subset,
        'buffer': 1.0,  # subset buffer [degree]
        'buffer_tracks': 5.0,  # subset buffer for track centers [degree]
        'lonv': lonv,
        'latv': latv,
        'draw_border': False,
        'draw_state': False,
        'central_longitude': central_longitude,
    }

    # Create a timedelta threshold
    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    dt_thres = datetime.timedelta(hours=1)

    # Data paths and filenames
    root_path = '/pscratch/sd/f/feng045/DYAMOND/'
    stats_path = f'{root_path}/mcs_stats/{phase}/{tracker}/'
    mask_path = f'{root_path}/mcs_mask/{phase}/{tracker}/'
    olr_path = f'{root_path}OLR_Precipitation_combined/'
    mcsstats_filebase = 'mcs_stats_'
    mcsmask_filebase = 'mcs_mask_'
    olr_filebase = 'olr_pcp_'
    trackstats_file = f"{stats_path}{mcsstats_filebase}{phase}_{runname}.nc"
    mask_file = f"{mask_path}{mcsmask_filebase}{phase}_{runname}.nc"
    olr_file = f"{olr_path}{olr_filebase}{phase}_{runname}.nc"

    pixel_radius = 10.0     # [km]

    # Output figure directory
    if out_dir is None:
        figdir = f'{pixeltracking_path}quicklooks_trackpaths/'
    else:
        figdir = f'{out_dir}/'
    os.makedirs(figdir, exist_ok=True)
    # Add to plot_info dictionary
    plot_info['figdir'] = figdir


    # Convert datetime string to Epoch time (base time)
    # These are for searching pixel-level files
    start_basetime = pd.to_datetime(start_datetime).timestamp()
    end_basetime = pd.to_datetime(end_datetime).timestamp()
    # Subtract start_datetime by TimeDelta to include tracks 
    # that start before the start_datetime but may not have ended yet
    TimeDelta = pd.Timedelta(days=4)
    start_datetime_4stats = (pd.to_datetime(start_datetime) - TimeDelta).strftime('%Y-%m-%dT%H')

    # Get track stats data
    # track_dict = get_track_stats(trackstats_file, start_datetime_4stats, end_datetime, dt_thres)
    track_dict = get_track_stats(trackstats_file, start_datetime_4stats, end_datetime, dt_thres, 
                                 map_extent=map_extent, buffer=map_info['buffer_tracks'])

    # Call function
    result = work_for_time_loop(olr_file, mask_file, track_dict, map_info, plot_info)

