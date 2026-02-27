### MODULES & PACKAGES

import os
import glob
import csv
import time
import copy
from datetime import date
from dateutil.relativedelta import relativedelta
import tempfile

import numpy as np
from scipy import ndimage
import xarray as xr
import iris
import iris.quickplot as qplt
import geopy.distance

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label

import warnings
warnings.filterwarnings("ignore")

### FILE PATHS, PARAMETERS, & TOGGLES

ar_dir = '/n/home10/ahatzius/AR_tracker/kennett_2021/SPCAM/' 
input_data_dir = '{}input_data/'.format(ar_dir)

testing = 0 #toggle for troubleshooting
if testing:
    years = [2000] 
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
else:
    years = list(range(1981, 2020))
    months = [1,2,3,4,5,6,7,8,9,10,11,12]

min_ivt = 600 #kg m**-1 s**-1
threshold_percentage = 80 
min_length = 1500
min_aspect = 2
min_span = 1000
min_size = 35

## filter toggles
use_filter_size = 1
use_filter_length = 1
use_filter_narrowness = 1
use_filter_meridional_ivt = 1
use_filter_ivt_coherence = 1
use_filter_orientation = 1
use_filter_longitude = 1
use_filter_latitude = 1

static_threshold = 1
threshold = 500

## generation toggles
include_ar_snapshot = 1
include_ar_nc_masks = 1
include_ar_char_csv = 1
include_filtering_data = 1

# --- daily aggregation (for lag/teleconnection work) ---
# If True, aggregate 6-hourly IVT into daily fields before AR detection.
# daily_method: 'sum' (recommended for daily-integrated transport) or 'mean'.
aggregate_daily = 0
daily_method = 'mean'
# If True, also write a daily summary CSV in addition to per-object rows.
include_daily_summary_csv = 1

## input data info
res = 1.5 #resolution
min_lat = 0.0
max_lat = 90.0
min_lon = 120.0
max_lon = 270.0

lat_grid_number = 48
lon_grid_number = 61

### DIRECTORY CREATION & CLEANUP

if not os.path.exists('{}ivt_thresholds'.format(ar_dir)):
    os.makedirs('ivt_thresholds')
if not os.path.exists('{}ar_snapshots'.format(ar_dir)):
    os.makedirs('ar_snapshots') # Contains AR .png snapshots
if not os.path.exists('{}ar_masks'.format(ar_dir)):
    os.makedirs('ar_masks') # Contains AR shape .nc masks
if not os.path.exists('{}ar_axes'.format(ar_dir)):
    os.makedirs('ar_axes') # Contains AR axes .nc masks
if not os.path.exists('{}ar_characteristics'.format(ar_dir)):
    os.makedirs('ar_characteristics') # Contains AR characteristics csv file
with open('{}ar_characteristics/ar_characteristics.csv'.format(ar_dir), 'a', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow([
    'year', 'month', 'day', 'time', 'label',
    'length_km', 'width_km',
    'mean_ivt', 'max_ivt',
    'mean_ivt_direction',
    'traj_bearing_deg',
    'surface_area_km2',
    'mean_latitude', 'min_latitude', 'max_latitude'
])
if not os.path.exists('{}filtering_data'.format(ar_dir)):
    os.makedirs('filtering_data') # Contains AR algorithm filtering csv
with open('{}filtering_data/filtering_data.csv'.format(ar_dir), 'a', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['year','month','initial','post_size','post_lat','post_longitude','post_length',
                        'post_narrowness','post_meridional_ivt','post_ivt_coherence',
                        'post_orientation'])

# Daily summary CSV (optional): counts + mean intensity/bearing for each day
if include_daily_summary_csv:
    if not os.path.exists(f'{ar_dir}daily_summary'):
        os.makedirs(f'{ar_dir}daily_summary')
    daily_summary_path = f'{ar_dir}daily_summary/daily_summary.csv'
    with open(daily_summary_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'year','month','day',
            'n_ar',
            'mean_mean_ivt','mean_max_ivt',
            'mean_traj_bearing'
        ])

import glob
for folder in ['ivt_thresholds', 'ar_snapshots', 'ar_masks', 'ar_axes']:
    for f in glob.glob(f'{ar_dir}{folder}/*'):
        os.remove(f)

### FUNCTIONS

def make_base_map():
    """set up a pacific-centered map projection and return an axis."""
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([min_lon, max_lon, 0, 80], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 15))

    # coastlines
    ax.coastlines(resolution='50m',
                    color='black',              
                    linewidth=0.5)
    return ax

def compute_valid_ids(i):
    """return list of surviving AR indices at timestep i after all filters."""
    all_filters = set(size_filter[i] + lat_filter[i] + lon_filter[i] +
                      filter_list_1[i] + filter_list_2[i] + filter_list_3[i] + 
                      filter_list_4[i] + filter_list_5[i])
    return [j for j in range(num_ob_list[i]) if (j + 1) not in all_filters]

def plot_snapshot(
    i, ar_dir, num_ob_list, size_filter, lat_filter,
    filter_list_1, filter_list_2, filter_list_3, filter_list_4, filter_list_5,
    ivt_frames, zero, labelled_object_list, axis_list, eastward_ivt_frames, northward_ivt_frames,
    axis_length_list, object_width_list, mean_ivt_magnitude_list, mean_ivt_direction_list
):
    """
    plot and save snapshots of atmospheric rivers for timestep i with fixed pacific-centered view.
    """
    valid_ids = compute_valid_ids(i)

    # extract time info
    frame = ivt_frames[i]
    time_coord = frame.coord('time')
    time_point = time_coord.units.num2date(time_coord.points[0])
    date_text = time_point.strftime('%Y-%m-%d')
    time_text = time_point.strftime('%H')
    print(f"Saving composite snapshot: date={date_text}, time={time_text}")
    # Hardcode Pacific-centered view for consistent framing
    AR_coords = (30, 180)
    print("Plot centered at fixed Pacific view:", AR_coords)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([120, 260, 0, 80], crs=ccrs.PlateCarree())


    # Manually define ticks and labels in °E / °N
    import matplotlib.ticker as mticker
    xticks = np.arange(120, 271, 30)
    xticks = np.sort(xticks)
    yticks = np.arange(0, 91, 15)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    # add lon/lat gridlines (no labels)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        xlocs=xticks,
        ylocs=yticks,
        linewidth=0.5,
        color='gray',
        alpha=0.7,
        linestyle='--',
        draw_labels=False,
    )
    # force fixed locators (avoids cartopy skipping some meridians like 210E/240E)
    import matplotlib.ticker as mticker
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)
    # ensure gridlines draw above background layers
    gl.xlines = True
    gl.ylines = True
    gl.zorder = 10

    # Manually set tick labels after gridlines (no duplicates elsewhere)
    ax.set_xticklabels([f"{int(x)}°E" for x in xticks], fontsize=8)
    ax.set_yticklabels([f"{int(y)}°N" for y in yticks], fontsize=8)

    # Add coastlines
    ax.coastlines(resolution='50m',
                    color='black',
                    linewidth=0.5)
    # Plot IVT contours (background)
    ivt_contour = (iris.plot.contourf(
        frame, levels=np.linspace(250,1500,11),
        cmap=matplotlib.cm.get_cmap('Blues'),
        zorder=0, extend='max'))
    # Plot IVT vectors
    uwind = eastward_ivt_frames[i]
    vwind = northward_ivt_frames[i]
    ulon = uwind.coord('longitude')
    vlon = vwind.coord('longitude')
    x = ulon.points
    y = vwind.coord('latitude').points
    u = uwind.data
    v = vwind.data
    plt.quiver(x[::12], y[::12], u[::12,::12], v[::12,::12],
                pivot='mid', scale = 45000, color='gray',
                zorder=5, width = 0.001)
    # AR boundary and axis overlays for all valid ARs
    colors2 = ["#FFFFFF00", "#20ff00"]
    cmap2 = matplotlib.colors.ListedColormap(colors2)
    colors1 = ["#FFFFFF00", "#FFFF00"]
    cmap1 = matplotlib.colors.ListedColormap(colors1)
    # Track number of ARs plotted for text box
    ar_count = 0
    for j in valid_ids:
        ar_count += 1
        # Plot AR boundary
        object_number = zero + (j+1)
        object_mask = (iris.analysis.maths.apply_ufunc(
                            np.equal,
                            labelled_object_list[i],
                            object_number))
        boundary_obj = object_mask.copy()
        iris.plot.contour(boundary_obj, levels=1, linewidths=0.5,
                            cmap=cmap2, zorder=20)
        # Plot AR axis (per AR, not global axis overlay)
        print(f"  plotting AR id={j+1} at timestep {i}")
        axis_mask = zero.copy()
        axis_mask.data = (axis_list[i].data == j + 1).astype(int)
        # use explicit discrete levels so only the axis (value==1) is filled
        iris.plot.contourf(axis_mask, levels=[0.5, 1.5], cmap=cmap1, zorder=20)
        # add a thin outline for clarity
        iris.plot.contour(axis_mask, levels=[1], linewidths=0.6, colors='yellow', zorder=21)
    # Plot colorbar
    cbar = plt.colorbar(ivt_contour, shrink = 0.55)
    cbar.set_label('IVT (kg/m/s)', rotation=270, labelpad = 10, fontsize=8)
    # Plot title
    plt.title("Atmospheric Rivers Detected at {}-{} UTC".format(date_text, time_text),
                fontdict = {'fontsize' : 10})
    # Compose summary stats for all ARs
    textstr = f"Date: {date_text} UTC\nNumber of ARs: {ar_count}\nObject Boundary: Green, Axis: Yellow"
    # These are matplotlib.patch.Patch properties
    props = dict(boxstyle='Square', facecolor='skyblue', alpha=0.3)
    # Place a text box
    ax.text(0, -0.15, textstr, transform=ax.transAxes, fontsize=6,
            verticalalignment='top', bbox=props)
    # Save Figure (one composite per timestep)
    plt.savefig('{}ar_snapshots/ar_{}-{}_composite.png'.format(ar_dir, date_text, time_text), dpi=700)
    plt.close("all")

def save_nc_masks(i, valid_ids, ivt_frames):
    """save .nc masks and axes for valid ARs."""
    time_coord = ivt_frames[i].coord('time')
    time_point = time_coord.units.num2date(time_coord.points[0])
    date_text = time_point.strftime('%Y-%m-%d')
    time_text = time_point.strftime('%H')
    for count, j in enumerate(valid_ids, start=1):
        mask = labelled_object_list[i].copy()
        mask.data = np.where(mask.data == j + 1, 1, 0)
        iris.save(mask, f'{ar_dir}ar_masks/ar_{date_text}-{time_text}_mask_{count}.nc')
        axis_mask = axis_list[i].copy()
        axis_mask.data = np.where(axis_mask.data == j + 1, 1, 0)
        iris.save(axis_mask, f'{ar_dir}ar_axes/ar_{date_text}-{time_text}_axes_{count}.nc')

def save_ar_csv(i, valid_ids, ivt_frames):
    """append AR characteristic rows to master csv."""
    time_coord = ivt_frames[i].coord('time')
    time_point = time_coord.units.num2date(time_coord.points[0])
    date_text = time_point.strftime('%Y-%m-%d')
    time_text = time_point.strftime('%H')
    year_str, month_str, day = date_text.split('-')
    time_str = time_text

    for count, j in enumerate(valid_ids, start=1):
        length = float(axis_length_list[i][j])
        width = float(object_width_list[i][j])
        mean_ivt = float(mean_ivt_magnitude_list[i][j])
        max_ivt = float(max_ivt_list[i][j]) if (i < len(max_ivt_list) and j < len(max_ivt_list[i])) else np.nan
        mean_ivt_dir = float(mean_ivt_direction_list[i][j])

        traj_bearing = float(traj_bearing_list[i][j]) if (i < len(traj_bearing_list) and j < len(traj_bearing_list[i])) else np.nan

        surface_area = float(object_area_list[i][j])

        blob_mask = (labelled_object_list[i].data == (j+1))
        lat_points = northward_ivt_list[i].coord('latitude').points
        blob_latitudes = lat_points[np.where(np.any(blob_mask, axis=1))]
        mean_latv = float(np.nanmean(blob_latitudes)) if blob_latitudes.size else np.nan
        min_latv = float(np.nanmin(blob_latitudes)) if blob_latitudes.size else np.nan
        max_latv = float(np.nanmax(blob_latitudes)) if blob_latitudes.size else np.nan

        row = [
            year_str, month_str, day, time_str, count,
            length, width,
            mean_ivt, max_ivt,
            mean_ivt_dir,
            traj_bearing,
            surface_area,
            mean_latv, min_latv, max_latv
        ]

        with open(f'{ar_dir}ar_characteristics/ar_characteristics.csv', 'a', newline='') as f:
            csv.writer(f).writerow(row)

def save_filter_stats():
    """summarize filtering numbers for current month."""
    num_size = [len(x) for x in size_filter]
    num_lat = [len(x) for x in lat_filter]
    num_long = [len(x) for x in lon_filter]
    num1 = [len(x) for x in filter_list_1]
    num2 = [len(x) for x in filter_list_2]
    num3 = [len(x) for x in filter_list_3]
    num4 = [len(x) for x in filter_list_4]
    num5 = [len(x) for x in filter_list_5]

    num_initial = np.sum(num_ob_list)
    post_size = num_initial - np.sum(num_size)
    post_lat = post_size - np.sum(num_lat)
    post_long = post_lat - np.sum(num_long)
    post_length = post_long - np.sum(num1)
    post_narrowness = post_length - np.sum(num2)
    post_meridional = post_narrowness - np.sum(num3)
    post_coherence = post_meridional - np.sum(num4)
    post_orientation = post_coherence - np.sum(num5)

    with open(f'{ar_dir}filtering_data/filtering_data.csv', 'a', newline='') as f:
        csv.writer(f).writerow([year, month, num_initial, post_size, post_lat, post_long,
                                post_length, post_narrowness, post_meridional,
                                post_coherence, post_orientation])

def compute_ar_axes(ivt, zero, labelled_object_list, ivt_list, ivt_direction_list, num_ob_list,
                    size_filter, northward_ivt_list, eastward_ivt_list):
    """Compute AR axes and trajectory bearing for each object."""
    from skimage.morphology import skeletonize

    axis_list = []
    axis_coords_list = []
    traj_bearing_list = []
    max_ivt_list = []

    template = zero.data
    n_timesteps = len(ivt_list)

    lats = northward_ivt_list[0].coord('latitude').points
    lons = _to_0_360(northward_ivt_list[0].coord('longitude').points)

    for i in range(n_timesteps):
        axis_data = np.zeros_like(template, dtype=np.int32)
        step_coords_list = []
        step_traj_bearing = []
        step_max_ivt = []

        n_objects = num_ob_list[i]
        ivt_frame = np.asarray(ivt_list[i].data, dtype=float)
        labelled_frame = labelled_object_list[i].data

        for j in range(n_objects):
            label_val = j + 1

            # skip filtered objects
            if label_val in size_filter[i]:
                step_coords_list.append([])
                step_traj_bearing.append(np.nan)
                step_max_ivt.append(np.nan)
                continue

            blob_mask = (labelled_frame == label_val)
            if not np.any(blob_mask):
                step_coords_list.append([])
                step_traj_bearing.append(np.nan)
                step_max_ivt.append(np.nan)
                continue

            blob_ivt = np.where(blob_mask, ivt_frame, np.nan)
            max_ivt = np.nanmax(blob_ivt)
            step_max_ivt.append(float(max_ivt))

            strong_mask = blob_mask & (blob_ivt > 0.7 * max_ivt)

            # Skeletonize to get axis
            skel = skeletonize(strong_mask.astype(np.uint8)).astype(bool)
            axis_data[skel] = label_val
            coords = np.argwhere(skel)
            step_coords_list.append([(np.array([r]), np.array([c])) for r, c in coords])

            # trajectory bearing from farthest skeleton endpoints
            ep1, ep2 = skeleton_endpoints_farthest(skel)
            if (ep1 is not None) and (ep2 is not None):
                r1, c1 = ep1
                r2, c2 = ep2
                step_traj_bearing.append(
                    float(bearing_deg(float(lats[r1]), float(lons[c1]), float(lats[r2]), float(lons[c2])))
                )
            else:
                step_traj_bearing.append(np.nan)

        axis_list.append(zero.copy(data=axis_data))
        axis_coords_list.append(step_coords_list)
        traj_bearing_list.append(step_traj_bearing)
        max_ivt_list.append(step_max_ivt)

    return (axis_list, axis_coords_list, traj_bearing_list, max_ivt_list)

def _to_0_360(lon):
    a = np.asarray(lon, float)
    return ((a + 360.) % 360.)

def bearing_deg(lat1, lon1, lat2, lon2):
    """Forward azimuth (degrees clockwise from North) from (lat1,lon1) to (lat2,lon2)."""
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    lam1 = np.deg2rad(lon1)
    lam2 = np.deg2rad(lon2)
    dlam = lam2 - lam1
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    theta = np.arctan2(x, y)
    return (np.rad2deg(theta) + 360) % 360

def skeleton_endpoints_farthest(skel_mask):
    """Return two (row,col) endpoints as the farthest pair of skeleton pixels."""
    coords = np.column_stack(np.nonzero(skel_mask))
    if coords.shape[0] < 2:
        return None, None
    diffs = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    i1, i2 = np.unravel_index(np.argmax(d2), d2.shape)
    return tuple(coords[i1]), tuple(coords[i2])

### DETECTION ALGORITHM

for year in years:
    for month in months:
        Date = date(year, month, 1)
        print('{}-{}'.format(year,month))

        ## LOAD DATA

        print('Loading Data')
        # load IVT data
        n_ivt_cubes = iris.load('{}n-ivt-{}-{}.nc'.format(input_data_dir,(Date).year, (Date).month))
        e_ivt_cubes = iris.load('{}e-ivt-{}-{}.nc'.format(input_data_dir,(Date).year, (Date).month))
        # assign cubes to IVT variables
        northward_ivt = n_ivt_cubes[0]
        eastward_ivt = e_ivt_cubes[0]
        lat_points = northward_ivt.coord("latitude").points
        lon_points = northward_ivt.coord("longitude").points

        # Helper to convert (row,col) to (lat,lon) using this month's cube coordinates
        def rc_to_latlon(rc):
            r = int(rc[0][0])
            c = int(rc[1][0])
            return float(lat_points[r]), float(lon_points[c])
        
        if use_filter_longitude:
            lon_points = northward_ivt.coord('longitude').points
            mask_lon = lon_points < 110
            northward_ivt.data[:, :, mask_lon] = np.nan
            eastward_ivt.data[:, :, mask_lon] = np.nan
        # compute magnitude
        ivt_mag = np.sqrt(northward_ivt.data**2 + eastward_ivt.data**2)
        ivt = northward_ivt.copy(data=ivt_mag)
        # compute direction
        ivt_dir = ivt.copy()
        ivt_dir.data = ((np.arctan2(eastward_ivt.data, northward_ivt.data) 
                               * 180 / np.pi) + 180) % 360
        
       ## SORT / AGGREGATE DATA

        ivt_list = []
        northward_ivt_list = []
        eastward_ivt_list = []
        ivt_direction_list = []

        time_coord_full = ivt.coord('time')
        time_datetimes = time_coord_full.units.num2date(time_coord_full.points)

        if aggregate_daily:
            # group timestep indices by UTC date
            by_date = {}
            for ti, dt in enumerate(time_datetimes):
                dkey = dt.strftime('%Y-%m-%d')
                by_date.setdefault(dkey, []).append(ti)

            # build daily cubes
            for dkey in sorted(by_date.keys()):
                idxs = by_date[dkey]
                mag_stack = ivt.data[idxs, :, :].astype(float)
                if daily_method.lower() == 'sum':
                    mag_daily = np.nansum(mag_stack, axis=0)
                else:
                    mag_daily = np.nanmean(mag_stack, axis=0)

                # daily mean components (direction diagnostics)
                n_stack = northward_ivt.data[idxs, :, :].astype(float)
                e_stack = eastward_ivt.data[idxs, :, :].astype(float)
                n_daily = np.nanmean(n_stack, axis=0)
                e_daily = np.nanmean(e_stack, axis=0)

                dir_daily = ((np.arctan2(e_daily, n_daily) * 180 / np.pi) + 180) % 360

                # create 2D cubes with a single time point (use first index's time)
                t0 = idxs[0]
                ivt_day = ivt[t0].copy(data=mag_daily)
                n_day = northward_ivt[t0].copy(data=n_daily)
                e_day = eastward_ivt[t0].copy(data=e_daily)
                dir_day = ivt_dir[t0].copy(data=dir_daily)

                ivt_list.append(ivt_day)
                northward_ivt_list.append(n_day)
                eastward_ivt_list.append(e_day)
                ivt_direction_list.append(dir_day)

            print(f'Aggregated to {len(ivt_list)} daily fields for {year}-{month:02d} using method={daily_method}')
        else:
            # Use all 6-hourly timesteps directly
            n_timesteps = ivt.shape[0]  # ✅ define before using
            for i in range(n_timesteps):
                ivt_list.append(ivt[i])
                northward_ivt_list.append(northward_ivt[i])
                eastward_ivt_list.append(eastward_ivt[i])
                ivt_direction_list.append(ivt_dir[i])
            print(f'Using {len(ivt_list)} 6-hourly timesteps for {year}-{month:02d}')

        # number of frames used downstream
        n_frames = len(ivt_list)
        zero = ivt_list[0].copy(data=np.zeros_like(ivt_list[0].data))

        ## COMPUTE IVT THRESHOLDs

        print('Loading Monthly IVT Climatology Thresholds')
        clim_file = '/n/home10/ahatzius/AR_tracker/kennett_2021/SPCAM/ivt_climatology_monthly.nc'
        ivt_clim_ds = xr.open_dataset(clim_file)

        # select the current month slice from climatology
        month_index = month  # assuming 1–12 indexing
        ivt_threshold_xr = ivt_clim_ds['IVT_climatology'].sel(month=month_index)
        thr2d = np.asarray(ivt_threshold_xr.values, dtype=float)

        print(f'Using climatological IVT threshold for month {month_index}')
        print(f'  Shape: {thr2d.shape}, Range: [{np.nanmin(thr2d):.1f}, {np.nanmax(thr2d):.1f}] kg/m/s')

        # subtract climatology to work entirely with IVT anomalies
        anomaly_ivt_list = []
        for cube in ivt_list:
            anomaly_ivt_list.append(cube.copy(data=np.asarray(cube.data, dtype=float) - thr2d))
        ivt_list = anomaly_ivt_list
        
        ## IDENTIFY OBJECTS

        print('Identifying Objects')

        labelled_object_list = []
        num_ob_list = []
        object_mask_list = []

        # Loop over frames in ivt_list (daily 2D cubes or 6-hourly 2D cubes)
        for i in range(len(ivt_list)):
            frame = ivt_list[i]
            
            # 1) Positive-anomaly mask
            M = (np.asarray(frame.data, dtype=float) > 0).astype(int)
            
            # 2) Strong-core mask (within-frame 95th percentile)
            ivt_array = np.array(frame.data, dtype=float, copy=True, order='C')
            ivt_95th = np.nanpercentile(ivt_array, 95)
            M95 = (np.asarray(frame.data, dtype=float) >= ivt_95th).astype(int)
            
            # 3) INTERSECTION: only keep pixels that are BOTH >0 AND ≥P95
            M_final = np.logical_and(M, M95).astype(int)
            
            # 4) Label connected components
            structure = np.ones((3, 3), dtype=int)
            labels, num = ndimage.label(M_final, structure=structure)
            
            # Store results
            mask_cube = frame.copy()
            mask_cube.data = M_final
            object_mask_list.append(mask_cube)
            
            labelled_object_list.append(frame.copy(data=labels))
            num_ob_list.append(int(num))

        ## PRELIMINARY CHECKS

        # size filter to speed up runtime
        size_filter = [[] for _ in range(len(ivt_list))]
        if use_filter_size:
            print('Size Criterion')
            for i in range(n_frames):
                Filter = []
                # Calculate sizes
                object_sizes = ndimage.sum(object_mask_list[i].data, 
                                           labelled_object_list[i].data, 
                                           list(range(1, num_ob_list[i]+1)))
                for j in range(num_ob_list[i]):
                    if object_sizes[j] > min_size:
                        size_check = True
                    else:
                        size_check = False
                    if size_check == False:
                        Filter.append(j+1)
                size_filter[i] = Filter

        # latitude filter: filter out objects where less than 85% of the blob is between 20° and 55° latitude
        lat_filter = [[] for _ in range(len(ivt_list))]
        print('Latitude Criterion')
        if use_filter_latitude:
            for i in range(n_frames):
                Filter = []
                latitudes = northward_ivt.coord('latitude').points
                for j in range(num_ob_list[i]):
                    if ((j+1) not in size_filter[i]):
                        # Get indices of blob cells
                        blob_mask = (labelled_object_list[i].data == (j+1))
                        # Count total blob cells
                        total_cells = np.sum(blob_mask)
                        # Get corresponding latitudes of each blob cell
                        lat_indices = np.where(blob_mask)
                        blob_lats = latitudes[lat_indices[0]]
                        # Count cells within [15, 65] degrees
                        valid_cells = np.sum((blob_lats >= 15) & (blob_lats <= 65))
                        if valid_cells / total_cells < 0.8:
                            Filter.append(j+1)
                lat_filter[i] = Filter

        # longitude filter: remove blobs with >=50% of cells between 110E and 150E
        lon_filter = [[] for _ in range(len(ivt_list))]
        if use_filter_longitude:
            print('Longitude Criterion')
            lon_points = northward_ivt.coord('longitude').points
            for i in range(n_frames):
                Filter = []
                for j in range(num_ob_list[i]):
                    if ((j+1) not in size_filter[i]):
                        blob_mask = (labelled_object_list[i].data == (j+1))
                        total_cells = np.sum(blob_mask)
                        if total_cells == 0:
                            continue
                        lon_indices = np.where(blob_mask)
                        blob_lons = lon_points[lon_indices[1]]
                        frac = np.sum((blob_lons >= 110) & (blob_lons <= 140)) / total_cells
                        if frac >= 0.5:
                            Filter.append(j+1)
                lon_filter[i] = Filter
        else:
            lon_filter = [[] for _ in range(len(ivt_list))]
        
        ## COMPUTE AXIS

        print('Computing AR Axes')
        (axis_list, axis_coords_list,
        traj_bearing_list, max_ivt_list) = compute_ar_axes(
            ivt, zero, labelled_object_list, ivt_list, ivt_direction_list,
            num_ob_list, size_filter,
            northward_ivt_list, eastward_ivt_list)
        
        ## COMPUTE AXIS LENGTH
        axis_length_list = []
        for i in range(n_frames):
            step_lengths = []
            for j in range(num_ob_list[i]):
                if (j+1) in size_filter[i] or (j+1) in lon_filter[i]:
                    step_lengths.append(0.0)
                    continue
                
                axis_coords = axis_coords_list[i][j]
                if len(axis_coords) < 2:
                    step_lengths.append(0.0)
                    continue
                
                length = 0.0
                for k in range(len(axis_coords) - 1):
                    lat1, lon1 = rc_to_latlon(axis_coords[k])
                    lat2, lon2 = rc_to_latlon(axis_coords[k+1])
                    length += geopy.distance.distance((lat1, lon1), (lat2, lon2)).km
                step_lengths.append(float(length))
            
            axis_length_list.append(step_lengths)

        ## CALCULATE SURFACE AREAS

        # Surface area is in square kilometres.
        object_area_list = []
        # Compute surface area of each grid cell
        grid_areas = zero.copy()
        grid_areas.coord('latitude').guess_bounds()
        grid_areas.coord('longitude').guess_bounds()
        grid_areas.data = (iris.analysis.cartography.area_weights(grid_areas) 
                          / (1000**2))
        # Compute surface area of Objects
        for i in range(n_frames):
            areas = ndimage.sum(grid_areas.data, 
                                   labelled_object_list[i].data, 
                                   list(range(1, num_ob_list[i]+1)))
            object_area_list.append(areas)

        ## CALCULATE WIDTHS

        # The Width of an Object is calculated as its surface area divided
        # by its length.
        object_width_list = []
        for i in range(n_frames):
            widths = []
            for j in range(num_ob_list[i]):
                if (axis_length_list[i][j] > 0):
                    width = object_area_list[i][j] / axis_length_list[i][j]
                    widths.append(width)
                else:
                    width = 0
                    widths.append(width)
            object_width_list.append(widths)
        
        ### AR CRITERIA ###
        ## LENGTH ##
        filter_list_1 = [[] for _ in range(n_frames)]
        if use_filter_length:
            print('Length Criterion')
            # Filter Objects based on axis length.
            for i in range(n_frames):
                Filter = []
                for j in range(num_ob_list[i]):
                    if ((j+1) not in size_filter[i]) and ((j+1) not in lat_filter[i]) and ((j+1) not in lon_filter[i]):
                        length_check = (axis_length_list[i][j] > min_length)
                        if length_check == False:
                            Filter.append(j+1)
                filter_list_1[i] = Filter
        
        ## NARROWNESS ##
        filter_list_2 = [[] for _ in range(n_frames)]
        if use_filter_narrowness:
            print('Narrowness Criterion')
            # Filter Objects based on length/width ratio.
            for i in range(n_frames):
                Filter = []
                for j in range(num_ob_list[i]):
                    if ((j+1) not in size_filter[i]) and ((j+1) not in lat_filter[i]) and ((j+1) not in lon_filter[i]) and ((j+1) not in filter_list_1[i]):
                        narrowness_check = ((axis_length_list[i][j] 
                                            / object_width_list[i][j]) > min_aspect)
                        if narrowness_check == False:
                            Filter.append(j+1)
                filter_list_2[i] = Filter

        ## MAX IVT COORDS
        max_IVT_coords_list = []
        for i in range(n_frames):
            NewList = []
            for j in range(num_ob_list[i]):
                if (len(axis_coords_list[i][j]) > 0):
                    coords_A = axis_coords_list[i][j][0]
                else:
                    coords_A = (0,0)
                NewList.append(coords_A)
            max_IVT_coords_list.append(NewList)

        ## MEANT IVT MAG
        mean_ivt_magnitude_list = []
        for i in range(n_frames):
            NewList = ndimage.mean(ivt_list[i].data, 
                                   labelled_object_list[i].data, 
                                   list(range(1, num_ob_list[i]+1)))
            mean_ivt_magnitude_list.append(NewList)

        ## MEAN IVT DIR
        mean_ivt_direction_list = []
        for i in range(n_frames):
            mean_northward_ivt = ndimage.mean(northward_ivt_list[i].data,
                                  labelled_object_list[i].data,
                                  list(range(1, num_ob_list[i]+1)))
            mean_eastward_ivt = ndimage.mean(eastward_ivt_list[i].data,
                                            labelled_object_list[i].data,
                                            list(range(1, num_ob_list[i]+1)))
            NewList = ((np.arctan2(mean_eastward_ivt, mean_northward_ivt) 
                        * 180 / np.pi) + 180) % 360
            mean_ivt_direction_list.append(NewList)

        ## OBJECT ORIENTATION
        object_orientation_list = []
        for i in range(n_frames):
            NewList = []
            for j in range(num_ob_list[i]):
                if len(axis_coords_list[i][j]) < 2:
                    NewList.append(np.nan)
                    continue
                
                coords_A = axis_coords_list[i][j][0]
                coords_B = axis_coords_list[i][j][-1]
                lat1, lon1 = rc_to_latlon(coords_A)
                lat2, lon2 = rc_to_latlon(coords_B)
                
                # bearing from A to B (toward), then convert to "from" convention
                bearing_toward = bearing_deg(lat1, lon1, lat2, lon2)
                orientation_from = (bearing_toward + 180.0) % 360.0
                NewList.append(float(orientation_from))
            object_orientation_list.append(NewList)

        ## AXIS DISTANCE (direct b/w start and end)
        axis_distance_list = []
        for i in range(n_frames):
            NewList = []
            for j in range(num_ob_list[i]):
                if len(axis_coords_list[i][j]) < 2:
                    NewList.append(0.0)
                    continue
                
                coords_A = axis_coords_list[i][j][0]
                coords_B = axis_coords_list[i][j][-1]
                lat1, lon1 = rc_to_latlon(coords_A)
                lat2, lon2 = rc_to_latlon(coords_B)
                
                axis_distance = geopy.distance.distance((lat1, lon1), (lat2, lon2)).km
                NewList.append(float(axis_distance))
            axis_distance_list.append(NewList)
        
        ## MEAN MERIDIONAL IVT ##
        filter_list_3 = [[] for _ in range(n_frames)]
        if use_filter_meridional_ivt:
            print('Meridional IVT Criterion')
            # An object is discarded if the mean IVT does not have a poleward 
            # component > 50 kg m**-1 s**-1.
            for i in range(n_frames):
                Filter = []
                for j in range(num_ob_list[i]):
                    if (((j+1) not in filter_list_1[i])
                        and ((j+1) not in size_filter[i])
                        and ((j+1) not in lat_filter[i])
                        and ((j+1) not in lon_filter[i])
                        and ((j+1) not in filter_list_2[i])):
                        mean_poleward_ivt = ndimage.mean(
                            northward_ivt_list[i].data,
                            labelled_object_list[i].data,
                            j + 1
                        )
                        mean_meridional_ivt_check = mean_poleward_ivt > 50
                        if mean_meridional_ivt_check == False:
                            Filter.append(j+1)
                filter_list_3[i] = Filter
        
        ## IVT DIRECTION COHERENCE ##
        filter_list_4 = [[] for _ in range(n_frames)]
        if use_filter_ivt_coherence:
            print('Coherence IVT Direction Criterion')
            # If more than half of the grid cells have IVT deviating more than 45°
            # from the object’s mean IVT, the object is filtered.
            for i in range(n_frames):
                Filter = []
                for j in range(num_ob_list[i]):
                    if (((j+1) not in size_filter[i])
                        and ((j+1) not in lat_filter[i])
                        and ((j+1) not in lon_filter[i])
                        and ((j+1) not in filter_list_1[i])
                        and ((j+1) not in filter_list_2[i])
                        and ((j+1) not in filter_list_3[i])):
                        mean_direction = mean_ivt_direction_list[i][j]
                        deviation_from_mean_direction = (iris.analysis.maths.apply_ufunc(
                            np.absolute, ivt_direction_list[i] - mean_direction))
                        percentage_coherence_ivt_direction = (ndimage.mean(
                            np.logical_or(deviation_from_mean_direction.data < 45, 
                                          deviation_from_mean_direction.data > 315), 
                            labelled_object_list[i].data, 
                            j+1))
                        coherence_ivt_check = percentage_coherence_ivt_direction > 0.5
                        if coherence_ivt_check == False:
                            Filter.append(j+1)
                filter_list_4[i] = Filter
            
        ## ORIENTATION-DIRECTION ##
        filter_list_5 = [[] for _ in range(n_frames)]
        if use_filter_orientation:
            print('Consistent Orientation Criterion')
            # If object orientation deviates from the mean IVT direction by more than 
            # 45 degrees, the object is filtered. Object orientation is calculated as
            # the angle between the first and last grid cells of the AR axis. Further,
            # the distance between the first and last grid cells of the AR axis must be
            # greater than 1000 km.
            for i in range(n_frames):
                Filter = []
                for j in range(num_ob_list[i]):
                    if (((j+1) not in size_filter[i])
                        and ((j+1) not in lat_filter[i])
                        and ((j+1) not in lon_filter[i])
                        and ((j+1) not in filter_list_1[i])
                        and ((j+1) not in filter_list_2[i])
                        and ((j+1) not in filter_list_3[i])
                        and ((j+1) not in filter_list_4[i])):
                        mean_direction = mean_ivt_direction_list[i][j]
                        object_orientation = object_orientation_list[i][j]
                        dev = np.abs(float(mean_direction) - float(object_orientation))
                        # handle 360° wrap (e.g. 5° vs 355° should be 10° apart, not 350°)
                        dev = min(dev, 360.0 - dev)

                        too_misaligned = dev > 45.0
                        too_short = axis_distance_list[i][j] < min_span

                        if too_misaligned or too_short:
                            Filter.append(j+1)
                filter_list_5[i] = Filter

        ### CREATE AR SNAPSHOT ###

        # main snapshot loop
        if include_ar_snapshot:
            print('Saving Snapshots')
            for i in range(n_frames):
                start = time.time()
                valid_ids = compute_valid_ids(i)
                if not valid_ids:
                    print(f"no ARs passed filters at timestep {i}")
                    continue
                else:
                    plot_snapshot(
                        i, ar_dir, num_ob_list,
                        size_filter, lat_filter,
                        filter_list_1, filter_list_2, filter_list_3, filter_list_4, filter_list_5,
                        ivt_list, zero, labelled_object_list, axis_list, eastward_ivt_list, northward_ivt_list,
                        axis_length_list, object_width_list, mean_ivt_magnitude_list, mean_ivt_direction_list)
                    print(f"  timestep {i+1}/{n_frames} completed in {time.time() - start:.2f}s")

        ### SAVE OUTPUTS ###
        print('Saving Outputs')

        # save .nc masks for valid objects
        if include_ar_nc_masks:
            for i in range(n_frames):
                valid_ids = compute_valid_ids(i)
                if valid_ids:
                    save_nc_masks(i, valid_ids, ivt_list)

        # save ar characteristics csv
        if include_ar_char_csv:
            for i in range(n_frames):
                valid_ids = compute_valid_ids(i)
                if valid_ids:
                    save_ar_csv(i, valid_ids, ivt_list)

        # save monthly filtering summary
        if include_filtering_data:
            save_filter_stats()

        # month summary + timing
        n_frames = len(ivt_list) if aggregate_daily else ivt.shape[0]
        total_month_survivors = sum(len(compute_valid_ids(i)) for i in range(n_frames))

        # write daily summary if requested and in daily mode
        if include_daily_summary_csv and aggregate_daily:
            for i in range(n_frames):
                time_coord = ivt_list[i].coord('time')
                time_point = time_coord.units.num2date(time_coord.points[0])
                dtext = time_point.strftime('%Y-%m-%d')
                yy, mm, dd = dtext.split('-')
                valid_ids = compute_valid_ids(i)
                if not valid_ids:
                    n_ar = 0
                    mu_mean = np.nan
                    mu_max = np.nan
                    mu_brg = np.nan
                else:
                    js = valid_ids
                    n_ar = len(js)
                    brg_vals = [traj_bearing_list[i][j] for j in js if not np.isnan(traj_bearing_list[i][j])]
                    mu_mean = float(np.nanmean([mean_ivt_magnitude_list[i][j] for j in js]))
                    mu_max = float(np.nanmean([max_ivt_list[i][j] for j in js]))
                    mu_brg = float(np.nanmean(brg_vals)) if brg_vals else np.nan

                with open(daily_summary_path, 'a', newline='') as f:
                    csv.writer(f).writerow([yy, mm, dd, n_ar, mu_mean, mu_max, mu_brg])
        print(f'=== {year}-{month:02d}: {total_month_survivors} ARs passed all filters ===')
