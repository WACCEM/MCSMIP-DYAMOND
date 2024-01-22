"""
Ploting MCS mask swaths and tracks during a specified period for a subset domain.

>python plot_subset_mcs_maskswath_mcsmip.py 
-s STARTDATE -e ENDDATE  
--extent lonmin lonmax latmin latmax (subset domain boundary)
--phase DYAMOND phase (Summer or Winter)
--runname DYAMOND data runname (OBS or model name)
--tracker MCS tracker name (PyFLEXTRKR, MOAAP, TOOCAN, etc.)
--subset 1

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
# import dask
# from dask.distributed import Client, LocalCluster
# import warnings
# warnings.filterwarnings("ignore")


#-----------------------------------------------------------------------
def parse_cmd_args():
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Plot MCS tracks on Tb & precipitation snapshots for a user-defined subset domain."
    )
    parser.add_argument("-s", "--start", help="first time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-e", "--end", help="last time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    # parser.add_argument("-c", "--config", help="yaml config file for tracking", required=False)
    # parser.add_argument("-p", "--parallel", help="flag to run in parallel (0:serial, 1:parallel)", type=int, default=0)
    # parser.add_argument("--n_workers", help="Number of workers for parallel", type=int, default=1)
    parser.add_argument("--extent", nargs='+', help="map extent (lonmin, lonmax, latmin, latmax)", type=float, default=None)
    parser.add_argument("--subset", help="flag to subset data (0:no, 1:yes)", type=int, default=1)
    parser.add_argument("--figsize", nargs='+', help="figure size (width, height) in inches", type=float, default=None)
    parser.add_argument("--output", help="ouput directory", default=None)
    # parser.add_argument("--figbasename", help="output figure base name", default="")
    # parser.add_argument("--trackstats_file", help="MCS track stats file name", default=None)
    # parser.add_argument("--pixel_path", help="Pixel-level tracknumer mask files directory", default=None)
    parser.add_argument("--phase", help="DYAMOND phase (Summer or Winter)", required=True)
    parser.add_argument("--runname", help="DYAMOND data runname (OBS or model name)", required=True)
    parser.add_argument("--tracker", help="MCS tracker name", required=True)
    parser.add_argument("--region", help="Subset region name", required=True)
    args = parser.parse_args()

    # Put arguments in a dictionary
    args_dict = {
        'start_datetime': args.start,
        'end_datetime': args.end,
        # 'run_parallel': args.parallel,
        # 'n_workers': args.n_workers,
        # 'config_file': args.config,
        'extent': args.extent,
        'subset': args.subset,
        'figsize': args.figsize,
        'out_dir': args.output,
        # 'figbasename': args.figbasename,
        # 'trackstats_file': args.trackstats_file,
        # 'pixeltracking_path': args.pixel_path,
        'phase': args.phase,
        'runname': args.runname,
        'tracker': args.tracker,
        'region': args.region,
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

#--------------------------------------------------------------------------
def prep_data(olr_file, mask_file, map_info, plot_info):

    map_extent = map_info.get('map_extent', None)
    buffer = map_info.get('buffer', 0)
    # Get plot info from dictionary
    perim_thick = plot_info.get('perim_thick')
    # Make fig names
    fdatetime = f'{start_datetime}_{end_datetime}'
    figname_mask = f'{figdir}{figbasename_mask}{fdatetime}.png'
    figname_count = f'{figdir}{figbasename_count}{fdatetime}.png'
    figname_cb = f'{figdir}colorbar_count_horizontal.png'
    
    # Read OLR/precipitation file, subset to input datetime range
    dso = xr.open_dataset(olr_file).sel(time=slice(start_datetime, end_datetime))

    # Read MCS mask file, subset to intput datetime range
    dsm = xr.open_dataset(mask_file).sel(time=slice(start_datetime, end_datetime))

    nt_dso = dso.dims['time']
    nt_dsm = dsm.dims['time']
        
    # Replace the lat/lon coordinates
    if nt_dso == nt_dsm:
        dsm = dsm.assign_coords({'time':dso['time'], 'lon':dso['lon'], 'lat':dso['lat']})
    else:
        print(f'ERROR: number of subset times not the same between OLR/PCP and MCS mask files!')
        print(f'Code will exit now.')
        sys.exit()

    if runname != 'OBS':
        # Convert OLR to Tb
        tb = olr_to_tb(dso['olr'])
        # Add Tb to DataSet
        dso['Tb'] = tb
        
    # Combine OLR/precipitation and MCS mask DataSets
    ds = xr.combine_by_coords([dsm, dso], combine_attrs='drop_conflicts')

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

    # Get data variables
    ny = ds_sub.sizes['lat']
    nx = ds_sub.sizes['lon']
    ntime = ds_sub.sizes['time']
    lon = ds_sub['lon']
    lat = ds_sub['lat']
    lat2d = ds_sub['latitude']
    lon2d = ds_sub['longitude']
    mcs_mask = ds_sub['mcs_mask'].load()

    # Get unique track numbers
    uniq_mcs_tracks, n_uniq_pix = np.unique(mcs_mask, return_counts=True)
    # Remove 0 & NaN (not a track)
    idx_valid = (uniq_mcs_tracks != 0) & (~np.isnan(uniq_mcs_tracks))
    uniq_mcs_tracks = uniq_mcs_tracks[idx_valid]
    n_uniq_pix = n_uniq_pix[idx_valid]
    ntracks_pix = len(uniq_mcs_tracks)

    # Make a dilation structure
    dilationstructure = make_dilation_structure(perim_thick, pixel_radius, pixel_radius)

    #------------------------------------------------------------------------
    # Get time accumulated swath and track center
    #------------------------------------------------------------------------
    mask_swath = np.full((ntracks_pix, ny, nx), 0, dtype=int)
    mask_perim = np.full((ntracks_pix, ny, nx), 0, dtype=int)
    count_swath = np.full((ny, nx), 0, dtype=int)
    tracks_lat = np.full((ntracks_pix, ntime), np.NaN, dtype=float)
    tracks_lon = np.full((ntracks_pix, ntime), np.NaN, dtype=float)
    # Loop over each track
    for ii in range(ntracks_pix):
        # Find the specific track mask, sum over time to get the swath
        iswath = (mcs_mask == uniq_mcs_tracks[ii]).sum(dim='time')
        count_swath[:,:] += iswath.data
        # Replace the swath area with a number
        mask_swath[ii,:,:] = iswath.where(iswath == 0, other=ii+1).data
        # Get swath perimeter
        mask_perim[ii,:,:] = label_perimeter(mask_swath[ii,:,:].squeeze(), dilationstructure)
        # Find specific track mask, average over space to get the mean lat/lon location time series
        tracks_lat[ii,:] = lat2d.where(mcs_mask == uniq_mcs_tracks[ii]).mean(dim=('lat','lon')).data
        tracks_lon[ii,:] = lon2d.where(mcs_mask == uniq_mcs_tracks[ii]).mean(dim=('lat','lon')).data

    plot_data = {
        'ntracks_pix': ntracks_pix,
        'ntime': ntime,
        'lon': lon,
        'lat': lat,
        'mask_swath': mask_swath,
        'mask_perim': mask_perim,
        'count_swath': count_swath, 
        'tracks_lat': tracks_lat,
        'tracks_lon': tracks_lon,
    }
    # Call plotting function
    fig = plot_map(plot_data, map_info, plot_info, figname=figname_mask, figtype='mask')
    plt.close(fig)
    fig = plot_map(plot_data, map_info, plot_info, figname=figname_count, figtype='count')
    plt.close(fig)
    fig = plot_colorbar(plot_data, plot_info, figname=figname_cb)
    plt.close(fig)
    # import pdb; pdb.set_trace()

    return

#--------------------------------------------------------------------------
def plot_colorbar(plot_data, plot_info, figname=None):
    """
    Plot a standalone colorbar
    """
    ntime = plot_data['ntime']
    # figsize = plot_info['figsize']
    dpi = plot_info['dpi']
    fontsize = plot_info['fontsize']
    cmap_count = plot_info['cmaps']['cmap_count']

    mpl.rcParams['font.size'] = fontsize*2
    mpl.rcParams['font.family'] = 'Helvetica'
    fig, ax = plt.subplots(figsize=(14, 1), layout='constrained', dpi=dpi)
    # Normalize colors
    cmap = plt.get_cmap(cmap_count)
    levels = np.arange(0, ntime, 1)
    cb_levels = None
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend='both')
    # Plot colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=cb_levels,
                        cax=ax, orientation='horizontal', label='Number of Hours')
    # cbar.cmap.set_under('white')
    fig.savefig(figname, dpi=dpi, facecolor='w', bbox_inches='tight')
    print(figname)
    return fig

#--------------------------------------------------------------------------
def plot_map(plot_data, map_info, plot_info, figname=None, figtype=None):
    """
    Plot a map with shaded colors and track paths.
    """
    ntracks_pix = plot_data['ntracks_pix']
    ntime = plot_data['ntime']
    lon = plot_data['lon']
    lat = plot_data['lat']
    mask_swath = plot_data['mask_swath']
    mask_perim = plot_data['mask_perim']
    count_swath = plot_data['count_swath']
    tracks_lat = plot_data['tracks_lat']
    tracks_lon = plot_data['tracks_lon']

    # Get plot info from dictionary
    cmap_mask = plot_info['cmaps']['cmap_mask']
    cmap_count = plot_info['cmaps']['cmap_count']
    mask_alpha = plot_info['mask_alpha']
    perim_alpha = plot_info['perim_alpha']
    titles = plot_info['titles'] 
    fontsize = plot_info['fontsize']
    marker_size = plot_info['marker_size']
    tracknumber_fontsize = plot_info['tracknumber_fontsize']
    trackpath_linewidth = plot_info['trackpath_linewidth']
    perim_thick = plot_info.get('perim_thick')
    map_edgecolor = plot_info['map_edgecolor']
    map_resolution = plot_info['map_resolution']
    figsize = plot_info['figsize']
    dpi = plot_info['dpi']
    # Map domain, lat/lon ticks, background map features
    map_extent = map_info['map_extent']
    lonv = map_info.get('lonv', None)
    latv = map_info.get('latv', None)
    draw_border = map_info.get('draw_border', False)
    draw_state = map_info.get('draw_state', False)
    central_longitude = map_info.get('central_longitude', None)

    #------------------------------------------------------------------------
    # Plotting
    #------------------------------------------------------------------------
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
    # gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1, 0.1])
    gs = gridspec.GridSpec(1, 1, height_ratios=[1], width_ratios=[1])
    gs.update(wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
    # Use GridSpecFromSubplotSpec for panel and colorbar
    # gs_cb = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], height_ratios=[1], width_ratios=[0.01,0.01], wspace=5)
    # ax1 = plt.subplot(gs[0], projection=proj)
    # cax1 = plt.subplot(gs_cb[0])

    ax1 = plt.subplot(gs[0], projection=proj)
    # Map elements
    ax1.set_extent(map_extent, crs=data_proj)
    ax1.add_feature(land, facecolor='none', edgecolor=map_edgecolor, zorder=4)
    if draw_border == True:
        ax1.add_feature(borders, edgecolor=map_edgecolor, facecolor='none', linewidth=0.8, zorder=4)
    if draw_state == True:
        ax1.add_feature(states, edgecolor=map_edgecolor, facecolor='none', linewidth=0.8, zorder=4)
    ax1.set_aspect('auto', adjustable=None)
    # Gridlines
    gl = ax1.gridlines(crs=data_proj, draw_labels=True, linestyle='--', linewidth=0.5)
    gl.right_labels = False
    gl.top_labels = False
    lonv = None
    latv = None
    if (lonv is not None) & (latv is not None):
        gl.xlocator = mpl.ticker.FixedLocator(lonv)
        gl.ylocator = mpl.ticker.FixedLocator(latv)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()        
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_title(titles, loc='left')

    # Set up mask colors
    cmap_m = plt.get_cmap(cmap_mask)
    # Set up color levels
    levels = np.arange(1, ntracks_pix+1, 1)
    # Normalize the color map
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap_m.N, clip=False)
    # Get the color for each level (for track line plots)
    lcolors = [cmap_m(norm(lev)) for lev in levels]

    # Set up count colors
    if (figtype == 'count'):
        cmap_c = plt.get_cmap(cmap_count)
        # Set up color levels
        levels = np.arange(0, ntime, 1)
        # Normalize the color map
        norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap_c.N, clip=False)
        # Plot count
        Zm = np.ma.masked_where(count_swath == 0, count_swath)
        pcm = ax1.pcolormesh(lon, lat, Zm, cmap=cmap_c, norm=norm, transform=data_proj)

    # Loop over each track
    for ii in range(ntracks_pix):
        if (figtype == 'mask'):
            # Plot swath mask for the track
            _swath = mask_swath[ii,:,:].squeeze()
            _perim = mask_perim[ii,:,:].squeeze()
            Zm = np.ma.masked_where(_swath==0, _swath)
            pcm = ax1.pcolormesh(lon, lat, Zm, cmap=cmap_m, norm=norm, transform=data_proj, alpha=mask_alpha)
            Zm = np.ma.masked_where(_perim==0, _perim)
            pcpr = ax1.pcolormesh(lon, lat, Zm, cmap=cmap_m, norm=norm, transform=data_proj, alpha=perim_alpha)

        # Plot track path
        ax1.plot(tracks_lon[ii,:], tracks_lat[ii,:], color=lcolors[ii], lw=trackpath_linewidth, transform=data_proj, zorder=10)
        marker_style = dict(edgecolor=lcolors[ii], facecolor=lcolors[ii], linestyle='-', marker='o')
        # Find the first index that is not-NaN
        s_idx = np.argmax(~np.isnan(tracks_lon[ii,:]))
        ax1.scatter(tracks_lon[ii,s_idx], tracks_lat[ii,s_idx], s=marker_size, zorder=10, transform=data_proj, **marker_style)
    
    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    canvas.print_png(figname)
    fig.savefig(figname)
    # fig.savefig(figname_mask, dpi=dpi, facecolor='w', bbox_inches='tight')
    print(figname)

    # import pdb; pdb.set_trace()
    return fig


if __name__ == "__main__":
    # Get the command-line arguments...
    args_dict = parse_cmd_args()
    start_datetime = args_dict.get('start_datetime')
    end_datetime = args_dict.get('end_datetime')

    map_extent = args_dict.get('extent')
    subset = args_dict.get('subset')
    figsize = args_dict.get('figsize')
    out_dir = args_dict.get('out_dir')
    # trackstats_file = args_dict.get('trackstats_file')
    phase = args_dict.get('phase')
    runname = args_dict.get('runname')
    tracker = args_dict.get('tracker')
    region = args_dict.get('region')
    figbasename_mask = f'{phase}_{runname}_{tracker}_{region}_swathmask_'
    figbasename_count = f'{phase}_{runname}_{tracker}_{region}_count_'

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
    cmap_mask = cc.cm['glasbey']
    cmap_count = cc.cm["CET_L4"]
    cmaps = {'cmap_mask': cmap_mask, 'cmap_count': cmap_count}
    titles = f'{runname} [{tracker}]'

    plot_info = {
        'cmaps': cmaps,
        'titles': titles,
        'fontsize': 10,
        'dpi': 300,
        'marker_size': 40,
        'trackpath_linewidth': 3,
        'tracknumber_fontsize': 12,
        'perim_thick': 10,  # [km]
        'mask_alpha': 0.4,
        'perim_alpha': 0.8,
        'map_edgecolor': 'k',
        'map_resolution': '50m',
        'map_central_lon': 180,
        'figsize': figsize,
        'figbasename_mask': figbasename_mask,
        'figbasename_count': figbasename_count,
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

    root_path = '/pscratch/sd/f/feng045/DYAMOND/'
    # stats_path = f'{root_path}/mcs_stats/{phase}/{tracker}/'
    mask_path = f'{root_path}/mcs_mask/{phase}/{tracker}/'
    olr_path = f'{root_path}OLR_Precipitation_combined/'
    mcsstats_filebase = 'mcs_stats_'
    mcsmask_filebase = 'mcs_mask_'
    olr_filebase = 'olr_pcp_'
    # trackstats_file = f"{stats_path}{mcsstats_filebase}{phase}_{runname}.nc"
    mask_file = f"{mask_path}{mcsmask_filebase}{phase}_{runname}.nc"
    olr_file = f"{olr_path}{olr_filebase}{phase}_{runname}.nc"

    pixel_radius = 10.0     # [km]

    # Output figure directory
    figdir = f'{out_dir}/'
    os.makedirs(figdir, exist_ok=True)
    # Add to plot_info dictionary
    plot_info['figdir'] = figdir

    # Call function
    result = prep_data(olr_file, mask_file, map_info, plot_info)


