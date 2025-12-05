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
    years = [1990] 
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
use_filter_landfall = 0
use_filter_size = 1
use_filter_length = 1
use_filter_narrowness = 1
use_filter_meridional_ivt = 1
use_filter_ivt_coherence = 1
use_filter_orientation = 1
use_filter_longitude = 0
use_filter_latitude = 1
use_filter_origin = 1  # backwards trajectory / ocean-origin filter

static_threshold = 1
threshold = 500

## generation toggles
include_ar_snapshot = 1
include_ar_nc_masks = 1
include_ar_char_csv = 1
include_filtering_data = 1

## input data info
res = 1.5 #resolution
min_lat = 0.0
max_lat = 90.0
min_lon = 120.0
max_lon = 270.0

lat_grid_number = 48
lon_grid_number = 61

## load land mask
land_mask_dir = f'{ar_dir}landmask_global.nc'
land_mask = iris.load(land_mask_dir)[0][0]
# rename coordinates for consistency
if 'lon' in [c.name() for c in land_mask.coords()]:
    land_mask.coord('lon').rename('longitude')
if 'lat' in [c.name() for c in land_mask.coords()]:
    land_mask.coord('lat').rename('latitude')
# extract to numpy array for flexible shape ops
lm = land_mask.data
if lm.ndim == 1:
    lm = np.tile(lm, (lat_grid_number, 1))  # replicate across latitude
land_mask_data = lm  # use this instead of land_mask.data later

### DIRECTORY CREATION & CLEANUP

if not os.path.exists('{}ivt_thresholds'.format(ar_dir)):
    os.makedirs('ivt_thresholds')
if not os.path.exists('{}ar_snapshots'.format(ar_dir)):
    os.makedirs('ar_snapshots') # Contains AR .png snapshots
if not os.path.exists('{}ar_masks'.format(ar_dir)):
    os.makedirs('ar_masks') # Contains AR shape .nc masks
if not os.path.exists('{}ar_axes'.format(ar_dir)):
    os.makedirs('ar_axes') # Contains AR axes .nc masks
if not os.path.exists('{}landfall_locations'.format(ar_dir)):
    os.makedirs('landfall_locations') # Contains max IVT cell .nc masks
if not os.path.exists('{}ar_characteristics'.format(ar_dir)):
    os.makedirs('ar_characteristics') # Contains AR characteristics csv file
with open('{}ar_characteristics/ar_characteristics.csv'.format(ar_dir), 'a', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['year', 'month', 'day', 'time', 'label', 'length', 'width',
                        'mean_ivt', 'mean_ivt_direction', 'landfall_ivt', 'landfall_ivt_direction',
                        'surface_area', 'mean_latitude', 'min_latitude', 'max_latitude'])
if not os.path.exists('{}filtering_data'.format(ar_dir)):
    os.makedirs('filtering_data') # Contains AR algorithm filtering csv
with open('{}filtering_data/filtering_data.csv'.format(ar_dir), 'a', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['year','month', 'initial','num_ob_landfall','num_ob_size','num_ob_lat','num_ob_long',
                        'num_ob_length','num_ob_narrowness', 'num_ob_max','num_ob_mean',
                        'num_ob_poleward_ivt','num_ob_ivt_coherence',
                        'num_ob_orientation'])
    
import glob
for folder in ['ivt_thresholds', 'ar_snapshots', 'ar_masks', 'ar_axes', 'landfall_locations']:
    for f in glob.glob(f'{ar_dir}{folder}/*'):
        os.remove(f)

### FUNCTIONS

def make_base_map():
    """set up a pacific-centered map projection and return an axis."""
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([120, 270, 0, 80], crs=ccrs.PlateCarree())

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
    all_filters = set(landfall_filter[i] + size_filter[i] + lat_filter[i] +
                      filter_list_1[i] + filter_list_2[i] + filter_list_3[i] + 
                      filter_list_4[i] + filter_list_5[i] + filter_list_6[i])
    return [j for j in range(num_ob_list[i]) if (j + 1) not in all_filters]

def plot_snapshot(
    i, ar_dir, num_ob_list, landfall_filter, size_filter, lat_filter,
    filter_list_1, filter_list_2, filter_list_3, filter_list_4, filter_list_5,
    ivt, zero, labelled_object_list, axis_list, eastward_ivt, northward_ivt,
    axis_length_list, object_width_list, mean_ivt_magnitude_list, mean_ivt_direction_list,
    landfall_ivt_magnitudes, landfall_ivt_directions
):
    """
    plot and save snapshots of atmospheric rivers for timestep i with fixed pacific-centered view.
    """
    valid_ids = compute_valid_ids(i)

    # extract time info
    time_coord = ivt[i].coord('time')
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
        ivt[i], levels=np.linspace(250,1500,11),
        cmap=matplotlib.cm.get_cmap('Blues'),
        zorder=0, extend='max'))
    # Plot IVT vectors
    uwind = eastward_ivt[i]
    vwind = northward_ivt[i]
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

def save_nc_masks(i, valid_ids):
    """save .nc masks, axes, and landfall for valid ARs."""
    time_coord = ivt[i].coord('time')
    time_point = time_coord.units.num2date(time_coord.points[0])
    date_text = time_point.strftime('%Y-%m-%d')
    time_text = time_point.strftime('%H')
    for count, j in enumerate(valid_ids, start=1):
        mask = labelled_object_list[i].copy()
        mask.data = np.where(mask.data == j + 1, 1, 0)
        iris.save(mask, f'{ar_dir}ar_masks/ar_{date_text}-{time_text}_mask_{count}.nc')
        iris.save(axis_list[i], f'{ar_dir}ar_axes/ar_{date_text}-{time_text}_axes_{count}.nc')
        iris.save(landfall_locations[i], f'{ar_dir}landfall_locations/ar_{date_text}-{time_text}_landfall_{count}.nc')

def save_ar_csv(i, valid_ids):
    """append AR characteristic rows to master csv."""
    time_coord = ivt[i].coord('time')
    time_point = time_coord.units.num2date(time_coord.points[0])
    date_text = time_point.strftime('%Y-%m-%d')
    time_text = time_point.strftime('%H')
    year_str, month_str, day = date_text.split('-')
    time_str = time_text

    for count, j in enumerate(valid_ids, start=1):
        length = str(round(axis_length_list[i][j], -1))[:-2]
        width = str(round(object_width_list[i][j], -1))[:-2]
        mean_ivt = str(round(mean_ivt_magnitude_list[i][j], 0))[:-2]
        mean_ivt_dir = str(round(mean_ivt_direction_list[i][j], 1))[:-2]
        landfall_ivt = str(round(landfall_ivt_magnitudes[i][j], 0))[:-2]
        landfall_dir = str(round(landfall_ivt_directions[i][j], 1))[:-2]
        surface_area = str(round(object_area_list[i][j], 1))

        blob_mask = (labelled_object_list[i].data == (j+1))
        lat_points = northward_ivt.coord('latitude').points
        blob_latitudes = lat_points[np.where(np.any(blob_mask, axis=1))]
        mean_lat = str(round(blob_latitudes.mean(), 2))
        min_lat = str(round(blob_latitudes.min(), 2))
        max_lat = str(round(blob_latitudes.max(), 2))

        row = [year_str, month_str, day, time_str, count, length, width,
            mean_ivt, mean_ivt_dir, landfall_ivt, landfall_dir,
            surface_area, mean_lat, min_lat, max_lat]

        with open(f'{ar_dir}ar_characteristics/ar_characteristics.csv', 'a', newline='') as f:
            csv.writer(f).writerow(row)

def save_filter_stats():
    """summarize filtering numbers for current month."""
    num_landfall = [len(x) for x in landfall_filter]
    num_size = [len(x) for x in size_filter]
    num_lat = [len(x) for x in lat_filter]

    num1 = [len(x) for x in filter_list_1]
    num2 = [len(x) for x in filter_list_2]

    num3 = [len(x) for x in filter_list_3]
    num4 = [len(x) for x in filter_list_4]
    num5 = [len(x) for x in filter_list_5]

    numbt = [len(x) for x in filter_list_6]

    num_ob_0 = np.sum(num_ob_list)
    num_ob_land = num_ob_0 - np.sum(num_landfall)
    num_ob_size = num_ob_land - np.sum(num_size)
    num_ob_lat = num_ob_size - np.sum(num_lat)

    num_ob_2 = num_ob_lat - np.sum(num1)
    num_ob_3 = num_ob_2 - np.sum(num2)
    num_ob_4 = num_ob_3 - np.sum(num3)
    num_ob_5 = num_ob_4 - np.sum(num4)
    num_ob_6 = num_ob_5 - np.sum(num5)
    num_ob_bt = num_ob_6 - np.sum(numbt)

    with open(f'{ar_dir}filtering_data/filtering_data.csv', 'a', newline='') as f:
        csv.writer(f).writerow([year, month, num_ob_0, num_ob_land,
                                num_ob_size, num_ob_lat, num_ob_2, num_ob_3, 
                                num_ob_4, num_ob_5, num_ob_6, num_ob_bt])

def compute_ar_axes(ivt, zero, labelled_object_list, ivt_list, ivt_direction_list, num_ob_list, size_filter, landfall_filter):
    """
    Compute AR axes, landfall locations, IVT magnitudes/directions, and shape classification for each AR object.
    Returns:
        (axis_list, axis_coords_list, landfall_locations, landfall_ivt_magnitudes, landfall_ivt_directions, shape_classification_list)
    """
    from skimage.morphology import skeletonize
    from skimage.graph import route_through_array
    import numpy as np
    def classify_blob_shape(blob_mask, filtered_blob_mask):
        # Classify the AR blob as 'zonal', 'meridional', 'recurve', or 'ambiguous'
        rows, cols = np.where(blob_mask)
        if len(rows) == 0:
            return 'ambiguous'
        lat_span = rows.max() - rows.min()
        lon_span = cols.max() - cols.min()
        if lat_span == 0 or lon_span == 0:
            return 'ambiguous'
        aspect = lon_span / lat_span if lat_span > 0 else np.inf
        if aspect > 2:
            return 'zonal'
        elif aspect < 0.5:
            return 'meridional'
        # For recurve: check if both spans are large enough and the blob is "curved"
        if lat_span > 10 and lon_span > 10:
            # crude check: if the skeleton's endpoints are not at the bounding box corners
            skel = skeletonize(filtered_blob_mask.astype(np.uint8))
            coords = np.argwhere(skel)
            if coords.shape[0] >= 2:
                end1 = coords[0]
                end2 = coords[-1]
                bbox_corners = np.array([[rows.min(), cols.min()], [rows.max(), cols.max()]])
                if not (np.allclose(end1, bbox_corners[0]) or np.allclose(end2, bbox_corners[1])):
                    return 'recurve'
        return 'ambiguous'

    axis_list = []
    axis_coords_list = []
    landfall_locations = []
    landfall_ivt_magnitudes = []
    landfall_ivt_directions = []
    shape_classification_list = []
    template = zero.data
    n_timesteps = len(ivt_list)
    for i in range(n_timesteps):
        axis_data = np.zeros_like(template, dtype=np.int32)
        landfall_data = np.zeros_like(template, dtype=np.int32)
        step_coords_list = []
        step_landfall_ivt_mags = []
        step_landfall_ivt_dirs = []
        step_shape_class = []
        n_objects = num_ob_list[i]
        ivt_frame = ivt_list[i].data
        dir_frame = ivt_direction_list[i].data
        labelled_frame = labelled_object_list[i].data
        for j in range(n_objects):
            label_val = j + 1
            # skip filtered objects
            if (label_val in size_filter[i]) or (label_val in landfall_filter[i]):
                step_coords_list.append([])
                step_landfall_ivt_mags.append(0)
                step_landfall_ivt_dirs.append(0)
                step_shape_class.append('ambiguous')
                continue
            blob_mask = (labelled_frame == label_val)
            if not np.any(blob_mask):
                step_coords_list.append([])
                step_landfall_ivt_mags.append(0)
                step_landfall_ivt_dirs.append(0)
                step_shape_class.append('ambiguous')
                continue
            blob_ivt = np.where(blob_mask, ivt_frame, 0)
            max_ivt = blob_ivt.max()
            strong_mask = blob_mask & (blob_ivt > 0.7 * max_ivt)
            # Classify blob shape
            shape_class = classify_blob_shape(blob_mask, strong_mask)
            step_shape_class.append(shape_class)
            # Skeletonize to get axis
            skel = skeletonize(strong_mask.astype(np.uint8))
            axis_data[skel] = label_val
            coords = np.argwhere(skel)
            step_coords_list.append([(np.array([r]), np.array([c])) for r, c in coords])
            # Landfall location (max IVT cell)
            r, c = np.unravel_index(np.argmax(blob_ivt), blob_ivt.shape)
            landfall_data[r, c] = label_val
            step_landfall_ivt_mags.append(ivt_frame[r, c])
            step_landfall_ivt_dirs.append(dir_frame[r, c])
        axis_cube = zero.copy(data=axis_data)
        landfall_cube = zero.copy(data=landfall_data)
        axis_list.append(axis_cube)
        axis_coords_list.append(step_coords_list)
        landfall_locations.append(landfall_cube)
        landfall_ivt_magnitudes.append(step_landfall_ivt_mags)
        landfall_ivt_directions.append(step_landfall_ivt_dirs)
        shape_classification_list.append(step_shape_class)
    return (axis_list, axis_coords_list, landfall_locations, landfall_ivt_magnitudes, landfall_ivt_directions, shape_classification_list)
                
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
        if use_filter_longitude:
            lon_points = northward_ivt.coord('longitude').points
            mask_lon = lon_points < 160
            northward_ivt.data[:, :, mask_lon] = np.nan
            eastward_ivt.data[:, :, mask_lon] = np.nan
        # compute magnitude
        ivt_mag = np.sqrt(northward_ivt.data**2 + eastward_ivt.data**2)
        ivt = northward_ivt.copy(data=ivt_mag)
        zero = 0 * ivt[0]
        # compute direction
        ivt_dir = ivt.copy()
        ivt_dir.data = ((np.arctan2(eastward_ivt.data, northward_ivt.data) 
                               * 180 / np.pi) + 180) % 360
        
        ## SORT DATA

        ivt_list = []
        northward_ivt_list = []
        eastward_ivt_list = []
        ivt_direction_list = []
        for i in range(ivt.shape[0]):
            ivt_list.append(ivt[i])
            northward_ivt_list.append(northward_ivt[i])
            eastward_ivt_list.append(eastward_ivt[i])
            ivt_direction_list.append(ivt_dir[i])

        ## COMPUTE IVT THRESHOLDs

        print('Loading Monthly IVT Climatology Thresholds')
        clim_file = '/n/home10/ahatzius/AR_tracker/kennett_2021/MERRA/ivt_climatology_monthly.nc'
        ivt_clim = iris.load_cube(clim_file)

        # select the current month slice from climatology
        month_index = month  # assuming 1–12 indexing
        ivt_threshold = ivt_clim.extract(iris.Constraint(month=month_index))

        print(f'Using climatological IVT threshold for month {month_index}')
        
        ## IDENTIFY OBJECTS

        print('Identifying Objects')

        object_mask = ivt.copy()
        object_mask.data = (ivt.data > ivt_threshold.data).astype(int)

        # create a second mask isolating only top 5% of IVT within the day
        ivt_array = np.array(ivt.data, dtype=float, copy=True, order='C')
        ivt_95th = np.percentile(ivt_array, 95)
        max_mask = ivt.copy()
        max_mask.data = (ivt.data >= ivt_95th).astype(int)

        # overlap max_mask with object_mask to refine candidate regions to keep only objects that intersect strong IVT cores
        refined_mask = ivt.copy()
        refined_mask.data = np.logical_and(object_mask.data, max_mask.data).astype(int)
        object_mask.data = refined_mask.data

        # label connected objects per timestep (8-connected)
        structure = np.ones((3, 3), dtype=int)
        labelled = np.empty_like(object_mask.data, dtype=int)

        labelled_object_list = []
        num_ob_list = []
        object_mask_list = []
        for i in range(ivt.shape[0]):
            object_mask_list.append(object_mask[i])

        for t in range(object_mask.shape[0]):
            labels, num = ndimage.label(object_mask.data[t], structure=structure)
            labelled[t] = labels

            label_cube = object_mask[t].copy(data=labels)
            labelled_object_list.append(label_cube)
            num_ob_list.append(num)

        ## PRELIMINARY CHECKS

        # landfall check; if enabled
        landfall_filter = [[] for _ in range(ivt.shape[0])]
        if use_filter_landfall:
            print('Landfall Criterion')
            for i in range(ivt.shape[0]):
                Filter = []
                Array = land_mask_data * labelled_object_list[i].data
                for j in range(num_ob_list[i]):
                    landfall_check = (j+1) in Array
                    if landfall_check == False:
                        Filter.append(j+1)
                landfall_filter[i] = Filter

        # size filter to speed up runtime
        size_filter = [[] for _ in range(ivt.shape[0])]
        if use_filter_size:
            print('Size Criterion')
            for i in range(ivt.shape[0]):
                Filter = []
                # Calculate sizes
                object_sizes = ndimage.sum(object_mask_list[i].data, 
                                           labelled_object_list[i].data, 
                                           list(range(1, num_ob_list[i]+1)))
                for j in range(num_ob_list[i]):
                    if ((j+1) not in landfall_filter[i]):
                        if object_sizes[j] > min_size:
                            size_check = True
                        else:
                            size_check = False
                        if size_check == False:
                            Filter.append(j+1)
                size_filter[i] = Filter

        # latitude filter: filter out objects where less than 85% of the blob is between 20° and 55° latitude
        lat_filter = [[] for _ in range(ivt.shape[0])]
        print('Latitude Criterion')
        if use_filter_latitude:
            for i in range(ivt.shape[0]):
                Filter = []
                latitudes = northward_ivt.coord('latitude').points
                for j in range(num_ob_list[i]):
                    if ((j+1) not in landfall_filter[i] and
                        (j+1) not in size_filter[i]):
                        # Get indices of blob cells
                        blob_mask = (labelled_object_list[i].data == (j+1))
                        # Count total blob cells
                        total_cells = np.sum(blob_mask)
                        # Get corresponding latitudes of each blob cell
                        lat_indices = np.where(blob_mask)
                        blob_lats = latitudes[lat_indices[0]]
                        # Count cells within [20, 55] degrees
                        valid_cells = np.sum((blob_lats >= 20) & (blob_lats <= 55))
                        if valid_cells / total_cells < 0.8:
                            Filter.append(j+1)
                lat_filter[i] = Filter
        
        ## COMPUTE AXIS

        print('Computing AR Axes')
        (axis_list, axis_coords_list, landfall_locations, landfall_ivt_magnitudes,
         landfall_ivt_directions, shape_classification_list) = compute_ar_axes(
            ivt, zero, labelled_object_list, ivt_list, ivt_direction_list,
            num_ob_list, size_filter, landfall_filter)
        
        ## COMPUTE AXIS LENGTH
        axis_length_list = []
        def calc_length(axis_coords):
            length = 0
            for k in list(range(len(axis_coords)-1)):
                lat1=max_lat-(res*np.abs(axis_coords[k][0]))
                lon1=min_lon+(res*np.abs(axis_coords[k][1]))
                lat2=max_lat-(res*np.abs(axis_coords[k+1][0]))
                lon2=min_lon+(res*np.abs(axis_coords[k+1][1]))
                coords_1 = (lat1, lon1)
                coords_2 = (lat2, lon2)
                segment = geopy.distance.distance(coords_1, coords_2).km
                length += segment
            return length
        for i in range(ivt.shape[0]):
            # Compute length of each Object
            step_length_list = []
            for j in range(num_ob_list[i]):
                if (((j+1) not in size_filter[i]) 
                    & ((j+1) not in landfall_filter[i])):
                    length = calc_length(axis_coords_list[i][j])
                    step_length_list.append(length)
                else:
                    step_length_list.append(0)
            axis_length_list.append(step_length_list) 

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
        for i in range(ivt.shape[0]):
            areas = ndimage.sum(grid_areas.data, 
                                   labelled_object_list[i].data, 
                                   list(range(1, num_ob_list[i]+1)))
            object_area_list.append(areas)

        ## CALCULATE WIDTHS

        # The Width of an Object is calculated as its surface area divided
        # by its length.
        object_width_list = []
        for i in range(ivt.shape[0]):
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
        filter_list_1 = [[] for _ in range(ivt.shape[0])]
        if use_filter_length:
            print('Length Criterion')
            # Filter Objects based on axis length.
            for i in range(ivt.shape[0]):
                Filter = []
                for j in range(num_ob_list[i]):
                    if ((j+1) not in landfall_filter[i]) and ((j+1) not in size_filter[i]) and ((j+1) not in lat_filter[i]):
                        length_check = (axis_length_list[i][j] > min_length)
                        if length_check == False:
                            Filter.append(j+1)
                filter_list_1[i] = Filter
        
        ## NARROWNESS ##
        filter_list_2 = [[] for _ in range(ivt.shape[0])]
        if use_filter_narrowness:
            print('Narrowness Criterion')
            # Filter Objects based on length/width ratio.
            for i in range(ivt.shape[0]):
                Filter = []
                for j in range(num_ob_list[i]):
                    if ((j+1) not in landfall_filter[i]) and ((j+1) not in size_filter[i]) and ((j+1) not in lat_filter[i]) and ((j+1) not in filter_list_1[i]):
                        narrowness_check = ((axis_length_list[i][j] 
                                            / object_width_list[i][j]) > min_aspect)
                        if narrowness_check == False:
                            Filter.append(j+1)
                filter_list_2[i] = Filter

        ## MAX IVT COORDS
        max_IVT_coords_list = []
        for i in range(ivt.shape[0]):
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
        for i in range(ivt.shape[0]):
            NewList = ndimage.mean(ivt_list[i].data, 
                                   labelled_object_list[i].data, 
                                   list(range(1, num_ob_list[i]+1)))
            mean_ivt_magnitude_list.append(NewList)

        ## MEAN IVT DIR
        mean_ivt_direction_list = []
        for i in range(ivt.shape[0]):
            mean_northward_ivt = ndimage.mean(northward_ivt[i].data, 
                                              labelled_object_list[i].data, 
                                              list(range(1, num_ob_list[i]+1)))
            mean_eastward_ivt = ndimage.mean(eastward_ivt[i].data, 
                                             labelled_object_list[i].data, 
                                             list(range(1, num_ob_list[i]+1)))
            NewList = ((np.arctan2(mean_eastward_ivt, mean_northward_ivt) 
                        * 180 / np.pi) + 180) % 360
            mean_ivt_direction_list.append(NewList)

        ## OBJECT ORIENTATION
        object_orientation_list = []
        for i in range(ivt.shape[0]):
            NewList = []
            for j in range(num_ob_list[i]):
                if (len(axis_coords_list[i][j]) > 0):
                    coords_A = axis_coords_list[i][j][0]
                    coords_B = axis_coords_list[i][j][-1]
                else:
                    coords_A = (0,0)
                    coords_B = (0,0)
                orientation = ((np.arctan2(coords_A[1] - coords_B[1], 
                                           coords_A[0] - coords_B[0]) 
                            * 180 / np.pi) + 180) % 360
                NewList.append(orientation)
            object_orientation_list.append(NewList)

        ## AXIS DISTANCE (direct b/w start and end)
        axis_distance_list = []
        for i in range(ivt.shape[0]):
            NewList = []
            for j in range(num_ob_list[i]):
                if (len(axis_coords_list[i][j]) > 0):
                    coords_A = axis_coords_list[i][j][0]
                    coords_B = axis_coords_list[i][j][-1]
                else:
                    coords_A = (0,0)
                    coords_B = (0,0)
                # BEGIN PATCH: clip latitudes to valid range for axis distance calc
                lat1 = np.clip(max_lat - (res * np.abs(coords_A[0])), -90, 90)
                lon1 = min_lon + (res * np.abs(coords_A[1]))
                lat2 = np.clip(max_lat - (res * np.abs(coords_B[0])), -90, 90)
                lon2 = min_lon + (res * np.abs(coords_B[1]))
                # END PATCH
                coords_1 = (lat1, lon1)
                coords_2 = (lat2, lon2)
                axis_distance = geopy.distance.distance(coords_1, coords_2).km
                NewList.append(axis_distance)
            axis_distance_list.append(NewList)
        
        ## MEAN MERIDIONAL IVT ##
        filter_list_3 = [[] for _ in range(ivt.shape[0])]
        if use_filter_meridional_ivt:
            print('Meridional IVT Criterion')
            # An object is discarded if the mean IVT does not have a poleward 
            # component > 50 kg m**-1 s**-1.
            for i in range(ivt.shape[0]):
                Filter = []
                for j in range(num_ob_list[i]):
                    if ((j+1) not in landfall_filter[i]
                        and ((j+1) not in filter_list_1[i]) 
                        and ((j+1) not in size_filter[i])
                        and ((j+1) not in lat_filter[i])
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
        filter_list_4 = [[] for _ in range(ivt.shape[0])]
        if use_filter_ivt_coherence:
            print('Coherence IVT Direction Criterion')
            # If more than half of the grid cells have IVT deviating more than 45°
            # from the object’s mean IVT, the object is filtered.
            for i in range(ivt.shape[0]):
                Filter = []
                for j in range(num_ob_list[i]):
                    if ((j+1) not in landfall_filter[i]
                        and ((j+1) not in size_filter[i])
                        and ((j+1) not in lat_filter[i])
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
        filter_list_5 = [[] for _ in range(ivt.shape[0])]
        if use_filter_orientation:
            print('Consistent Orientation Criterion')
            # If object orientation deviates from the mean IVT direction by more than 
            # 45 degrees, the object is filtered. Object orientation is calculated as
            # the angle between the first and last grid cells of the AR axis. Further,
            # the distance between the first and last grid cells of the AR axis must be
            # greater than 1000 km.
            for i in range(ivt.shape[0]):
                Filter = []
                for j in range(num_ob_list[i]):
                    if ((j+1) not in landfall_filter[i]
                        and ((j+1) not in size_filter[i])
                        and ((j+1) not in lat_filter[i])
                        and ((j+1) not in filter_list_1[i])
                        and ((j+1) not in filter_list_2[i])
                        and ((j+1) not in filter_list_3[i])
                        and ((j+1) not in filter_list_4[i])):
                        mean_direction = mean_ivt_direction_list[i][j]
                        object_orientation = object_orientation_list[i][j]
                        deviation_from_mean_direction = np.absolute(float(mean_direction)- 
                                                                    float(object_orientation))
                        consistent_orientation_check = deviation_from_mean_direction > 45
                        axis_distance_check = axis_distance_list[i][j] < min_span
                        if (consistent_orientation_check or axis_distance_check) == False:
                            Filter.append(j+1)
                filter_list_5[i] = Filter

        ## ORIGIN / BACKWARD TRAJECTORY FILTER ##
        filter_list_6 = [[] for _ in range(ivt.shape[0])]
        if use_filter_origin:
            print("Origin Direction Criterion (Backwards Trajectory)")
            # Filters ARs whose backward IVT streamline doesn't reach open ocean or intersects Asia
            max_back_distance_km = 2000  # maximum distance to trace back
            step_size_km = 100           # step size (~100 km per step)
            max_steps = int(max_back_distance_km / step_size_km)

            lats = northward_ivt.coord('latitude').points
            lons = northward_ivt.coord('longitude').points

            for i in range(ivt.shape[0]):
                Filter = []
                for j in range(num_ob_list[i]):
                    # skip objects already filtered out
                    if ((j+1) in landfall_filter[i]
                        or (j+1) in size_filter[i]
                        or (j+1) in lat_filter[i]
                        or (j+1) in filter_list_1[i]
                        or (j+1) in filter_list_2[i]
                        or (j+1) in filter_list_3[i]
                        or (j+1) in filter_list_4[i]
                        or (j+1) in filter_list_5[i]):
                        continue

                    coords = axis_coords_list[i][j]
                    if not coords:
                        Filter.append(j+1)
                        continue

                    # flatten coords list and find westernmost pixel (handle both scalars and 1-element arrays)
                    flat_coords = np.array([[np.ravel(r)[0], np.ravel(c)[0]] for (r, c) in coords if len(np.ravel(r)) and len(np.ravel(c))])
                    western_idx = np.argmin(flat_coords[:, 1])
                    r = int(flat_coords[western_idx, 0])
                    c = int(flat_coords[western_idx, 1])

                    # initialize position
                    lat = lats[r]
                    lon = lons[c]
                    success = False

                    for step in range(max_steps):
                        u = eastward_ivt[i].data[r, c]
                        v = northward_ivt[i].data[r, c]
                        mag = np.sqrt(u**2 + v**2)
                        if mag == 0 or np.isnan(mag):
                            break

                        # move opposite IVT vector direction
                        dlat = -(v / mag) * (step_size_km / 111.0)
                        dlon = -(u / mag) * (step_size_km / (111.0 * np.cos(np.deg2rad(lat))))
                        lat += dlat
                        lon += dlon

                        # check domain bounds
                        if (lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon):
                            break

                        # Check for intersection with Asian landmass (lat 10–80N, lon 60–150E)
                        if (10 <= lat <= 80) and (60 <= lon <= 150):
                            Filter.append(j+1)
                            success = False
                            break

                        # find nearest grid indices
                        r = np.argmin(np.abs(lats - lat))
                        c = np.argmin(np.abs(lons - lon))

                        # check if ocean cell (assuming 0 = ocean, 1 = land)
                        if land_mask_data[r, c] == 0:
                            success = True
                            break

                    if not success and (j+1) not in Filter:
                        Filter.append(j+1)

                filter_list_6[i] = Filter
        
        ### CREATE AR SNAPSHOT ###

        # main snapshot loop
        if include_ar_snapshot:
            print('Saving Snapshots')
            for i in range(ivt.shape[0]):
                start = time.time()
                valid_ids = compute_valid_ids(i)
                if not valid_ids:
                    print(f"no ARs passed filters at timestep {i}")
                    continue
                else:
                    plot_snapshot(
                        i, ar_dir, num_ob_list,
                        landfall_filter, size_filter, lat_filter,
                        filter_list_1, filter_list_2, filter_list_3, filter_list_4, filter_list_5,
                        ivt, zero, labelled_object_list, axis_list, eastward_ivt, northward_ivt,
                        axis_length_list, object_width_list, mean_ivt_magnitude_list, mean_ivt_direction_list,
                        landfall_ivt_magnitudes, landfall_ivt_directions)
                    print(f"  timestep {i+1}/{ivt.shape[0]} completed in {time.time() - start:.2f}s")

        ### SAVE OUTPUTS ###
        print('Saving Outputs')

        # save .nc masks for valid objects
        if include_ar_nc_masks:
            for i in range(ivt.shape[0]):
                valid_ids = compute_valid_ids(i)
                if valid_ids:
                    save_nc_masks(i, valid_ids)

        # save ar characteristics csv
        if include_ar_char_csv:
            for i in range(ivt.shape[0]):
                valid_ids = compute_valid_ids(i)
                if valid_ids:
                    save_ar_csv(i, valid_ids)

        # save monthly filtering summary
        if include_filtering_data:
            save_filter_stats()

        # month summary + timing
        total_month_survivors = sum(len(compute_valid_ids(i)) for i in range(ivt.shape[0]))
        print(f'=== {year}-{month:02d}: {total_month_survivors} ARs passed all filters ===')
