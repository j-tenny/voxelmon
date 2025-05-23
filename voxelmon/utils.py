import numpy as np
import pandas as pd
from typing import Union, Sequence, Tuple


def get_files_list(directory, keyword, recursive=True):
    import os
    import fnmatch
    matching_files = []
    if recursive:
        for root, dirs, files in os.walk(directory):
            for filename in fnmatch.filter(files, f'*{keyword}*'):
                matching_files.append(os.path.join(root, filename))
    else:
        files = os.listdir(directory)
        for filename in fnmatch.filter(files, f'*{keyword}*'):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                matching_files.append(path)
    return matching_files


def directory_to_pandas(directory,keyword='.csv',recursive=False,filename_col:Union[str,None]='filename',include_filename_ext=False):
    """
    Read csv files from a directory and concatenate in a pandas dataframe
    Args:
        directory: path to dir
        keyword: keyword used in filtering files
        recursive: recursively search through directories within directory
        filename_col: name of column to include filenames. If None, does not append.
        include_filename_ext: include filename extension in filenames column

    Returns:

    """
    import os
    files = get_files_list(directory,keyword,recursive)
    results = []
    for file in files:
        df = pd.read_csv(file)
        if filename_col is not None:
            if include_filename_ext:
                df['filename'] = os.path.basename(file)
            else:
                df['filename'] = os.path.splitext(os.path.basename(file))[0]
        results.append(df)
    return pd.concat(results).reset_index(drop=True)


def bin2D(pulses,function,cellSize,asArray = True,binExtents=None):
    # Function should be from polars and should specify a column name x, y, or z, e.g. pl.min('z')
    # binExtents must be iterable formatted as [xBinMin,yBinMin,xBinMax,yBinMax] noninclusive of end
    # if binExtents is None, extents are automatically pulled from pulses extents, may need to clip first
    import polars as pl
    import numpy as np
    try:
        points_df = pl.DataFrame({'x':pulses.xyz[:,0],'y':pulses.xyz[:,1],'z':pulses.xyz[:,2]})
    except:
        points_df = pl.DataFrame({'x':pulses[:,0],'y':pulses[:,1],'z':pulses[:,2]})

    points_df = points_df.with_columns(pl.col('x').floordiv(cellSize).cast(pl.Int32).alias('xBin'),
                                       pl.col('y').floordiv(cellSize).cast(pl.Int32).alias('yBin'))

    if binExtents is None:
        binMin = points_df[:,-2:].min().to_numpy().flatten()
        binMax = points_df[:, -2:].max().to_numpy().flatten()+1
        binExtents = np.concatenate([binMin,binMax])

    yBins,xBins = np.meshgrid(np.arange(binExtents[1],binExtents[3]),np.arange(binExtents[0],binExtents[2]),indexing='ij')

    bins_df = pl.DataFrame({'xBin': xBins.flatten().astype(np.int32),
                            'yBin': yBins.flatten().astype(np.int32)})

    binVals = points_df.drop_nulls().group_by(['xBin','yBin']).agg(function)

    result = bins_df.join(binVals,['xBin','yBin'],'left')
    if asArray:
        return result[:, -1].to_numpy().reshape(binExtents[2]-binExtents[0],binExtents[3]-binExtents[1],order='f')
    else:
        return result


def bin3D(pulses, function, cellSize,asArray = True, binExtents=None):
    # Function should be from polars and should specify a column name x, y, or z, e.g. pl.min('z')
    # binExtents must be iterable formatted as [xBinMin,yBinMin,zBinMin,xBinMax,yBinMax,zBinMax] noninclusive of end
    # if binExtents is None, extents are automatically pulled from pulses extents, may need to clip first
    import polars as pl
    import numpy as np
    if hasattr(cellSize,'__iter__'):
        cellSize = np.array(cellSize)
        if cellSize.size != 3:
            raise ValueError('cellSize must be scalar or 3 dimensional')
    else:
        cellSize = np.repeat(cellSize,3)

    try:
        points_df = pl.DataFrame({'x': pulses.xyz[:, 0], 'y': pulses.xyz[:, 1], 'z': pulses.xyz[:, 2]})
    except:
        points_df = pl.DataFrame({'x': pulses[:, 0], 'y': pulses[:, 1], 'z': pulses[:, 2]})

    points_df = points_df.with_columns(pl.col('x').floordiv(cellSize[0]).cast(pl.Int32).alias('xBin'),
                                       pl.col('y').floordiv(cellSize[1]).cast(pl.Int32).alias('yBin'),
                                       pl.col('z').floordiv(cellSize[2]).cast(pl.Int32).alias('zBin'))

    if binExtents is None:
        binMin = points_df[:, -3:].min().to_numpy().flatten()
        binMax = points_df[:, -3:].max().to_numpy().flatten() + 1
        binExtents = np.concatenate([binMin, binMax])

    zBins, yBins, xBins = np.meshgrid(np.arange(binExtents[2], binExtents[5]),
                                      np.arange(binExtents[1], binExtents[4]),
                                      np.arange(binExtents[0], binExtents[3]), indexing='ij')

    gridShape = binExtents[3:6]-binExtents[0:3]

    bins_df = pl.DataFrame({'xBin': xBins.flatten().astype(np.int32),
                            'yBin': yBins.flatten().astype(np.int32),
                            'zBin': zBins.flatten().astype(np.int32)})

    binVals = points_df.drop_nulls().group_by(['xBin', 'yBin','zBin']).agg(function)

    result = bins_df.join(binVals, ['xBin', 'yBin','zBin'], 'left').sort(['zBin','yBin','xBin'])
    if asArray:
        return result[:, -1].to_numpy().reshape(gridShape,order='f')
    else:
        return result

def open_file_pdal(filepath,bounds=None,calculate_height=True)->Tuple['pl.DataFrame',str]:
    """Read a file to a polars dataframe with pdal. Returns pl.DataFrame and crs

    Args:
        filepath (str): Path to ALS file readable by pdal. Type is inferred by extension.
        bounds (str): Clip extents of the resource in 2 or 3 dimensions, formatted as pdal-compatible string,
            e.g.: ([xmin, xmax], [ymin, ymax], [zmin, zmax]). If omitted, the entire dataset will be selected.
            The bounds can be followed by a slash (‘/’) and a spatial reference specification to apply to the bounds.
                """
    import pdal
    import polars as pl

    result=0
    if calculate_height:
        count = 10
        while result == 0 and count < 100:
            try:
                if bounds is not None:
                    pipeline = pdal.Pipeline([pdal.Reader(filepath, bounds=bounds),
                                              pdal.Filter.hag_delaunay(count=count)])
                else:
                    pipeline = pdal.Pipeline([pdal.Reader(filepath), pdal.Filter.hag_delaunay(count=count)])
                result = pipeline.execute()
            except:
                count *= 2
    else:
        if bounds is not None:
            pipeline = pdal.Pipeline([pdal.Reader(filepath, bounds=bounds)])
        else:
            pipeline = pdal.Pipeline([pdal.Reader(filepath)])
        pipeline.execute()

    return pl.DataFrame(pipeline.arrays[0]), pipeline.srswkt2


def is_noise_ivf(pulses,voxelSize=1,windowSize=3,minPointCount=200):
    from scipy import ndimage
    import polars as pl
    import numpy as np
    try:
        points_df = pl.DataFrame({'x': pulses.xyz[:, 0], 'y': pulses.xyz[:, 1], 'z': pulses.xyz[:, 2]})
    except:
        points_df = pl.DataFrame({'x': pulses[:, 0], 'y': pulses[:, 1], 'z': pulses[:, 2]})

    points_df = points_df.with_columns(pl.col('x').floordiv(voxelSize).cast(pl.Int32).alias('xBin'),
                                       pl.col('y').floordiv(voxelSize).cast(pl.Int32).alias('yBin'),
                                       pl.col('z').floordiv(voxelSize).cast(pl.Int32).alias('zBin'))

    # Get count of returns in voxels
    df = bin3D(pulses,function=pl.len(),cellSize=voxelSize,asArray=False)
    binMin = df[:, :3].min().to_numpy().flatten()
    binMax = df[:, :3].max().to_numpy().flatten() + 1
    gridShape = binMax-binMin
    arr = df[:,-1].to_numpy().reshape(gridShape,order='f')

    # Fill missing values as 0
    arr[np.isnan(arr)]=0
    # Sum count of points within moving window
    sum = ndimage.uniform_filter(arr.astype(np.float32),windowSize) * windowSize**2
    sum[np.isnan(sum)]=0
    # Mark voxels with few neighbors as noise
    noise = sum<=minPointCount
    # Append column to df with bins
    df = df.with_columns(pl.Series(noise.flatten('f')).alias('is_noise'))
    # Join with points, return boolean array for points where true signifies noise
    return points_df.join(df,('xBin','yBin','zBin'),'left').select('is_noise').to_numpy().flatten()

def normalize(xyz_df,dem_df,cellSize=None):
    import polars as pl
    if cellSize is None:
        cellSize = dem_df[1,0]-dem_df[0,0]

    dem_df = dem_df.rename({'z':'z_dem'})
    dem_df = dem_df.drop_nulls()

    xyz_df = xyz_df.with_columns(pl.col('x').floordiv(cellSize).cast(pl.Int32).alias('xBinDEM'),
                                 pl.col('y').floordiv(cellSize).cast(pl.Int32).alias('yBinDEM'))
    dem_df = dem_df.with_columns(pl.col('x').floordiv(cellSize).cast(pl.Int32).alias('xBinDEM'),
                                 pl.col('y').floordiv(cellSize).cast(pl.Int32).alias('yBinDEM'))
    xyz_df = xyz_df.join(dem_df,on=['xBinDEM','yBinDEM'],how='left')
    xyz_df = xyz_df.with_columns(pl.col('z').sub(pl.col('z_dem')).alias('z'))
    xyz_df = xyz_df.drop(['z_dem','xBinDEM','yBinDEM'])
    xyz_df = xyz_df.filter((pl.col('z').is_not_nan()) & (pl.col('z').is_not_null()))
    return xyz_df


def plot_side_view(xyz,direction=0,demPtsNormalize=None,returnData=False):
    # dir=0=+y, dir=1=+x, dir=2=-y, dir=3=-x
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt

    points_df = pl.DataFrame({'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2]})
    if demPtsNormalize is not None:
        points_df = normalize(points_df,demPtsNormalize)
        points_df = points_df.filter(pl.col('z')>=0)

    mincoords = points_df.select(['x', 'y', 'z']).min().to_numpy().flatten()
    maxcoords = points_df.select(['x', 'y', 'z']).max().to_numpy().flatten()

    extents3D = np.concatenate([mincoords,maxcoords])

    if direction == 0:
        bins = bin3D(points_df, function=pl.min('y'), cellSize=.1, asArray=True)
        bins = np.nanmin(bins,axis=1)
        extents2D = extents3D[[0,3,2,5]]
    elif direction == 1:
        bins = bin3D(points_df, function=pl.min('x'), cellSize=.1, asArray=True)
        bins = np.nanmin(bins, axis=0)
        extents2D = extents3D[[1, 4, 2, 5]]
    elif direction == 2:
        bins = bin3D(points_df, function=pl.max('y'), cellSize=.1, asArray=True)
        bins = np.nanmax(bins, axis=1)
        extents2D = extents3D[[0, 3, 2, 5]]
    else:
        bins = bin3D(points_df, function=pl.max('x'), cellSize=.1, asArray=True)
        bins = np.nanmax(bins, axis=0)
        extents2D = extents3D[[1, 4, 2, 5]]

    if returnData:
        return [np.rot90(bins),extents2D]
    else:
        return plt.imshow(np.rot90(bins),extent=extents2D)

def summarize_profiles(profiles, bin_height=.1, min_height=1., fsg_threshold=.011, cbd_col='cbd_pred', height_col ='height', pad_col='pad', occlusion_col='occluded', plot_id_col='Plot_ID'):

    profiles_filter_h = profiles[profiles[height_col]>=min_height]
    summary = profiles_filter_h.pivot_table(index=plot_id_col,
                                            aggfunc={cbd_col: 'sum', pad_col: 'sum', occlusion_col: 'mean'}
                                            ).reset_index()

    summary.columns = [plot_id_col,'cfl','occlusion','plant_area']
    summary['cfl'] *= bin_height
    summary['plant_area'] *= bin_height
    summary['fsg'] = 0.
    summary['fsg_h1'] = float(min_height)
    summary['fsg_h2'] = float(min_height)
    summary['canopy_height'] = 0.
    summary['canopy_ratio'] = 0.
    summary.insert(1,'cbd_max',0.)

    # Estimate cbd and fuel strata gap. Tolerate noise in profile by looking for a block of bins below the fsg threshold.
    min_contiguous_empty = .5 # Meters
    min_contiguous_filled = .5 # Meters
    #noise_bumps_allowed = 2 # Number of noisy bins
    noise_bumps_allowed = int(.25 * (min_contiguous_empty / bin_height)) + 1 # Number of noisy bins

    for plot_id in profiles[plot_id_col].unique():
        profile = profiles[profiles[plot_id_col]==plot_id]
        profile = profile.sort_values(by=height_col)
        profile = profile[[height_col,cbd_col]].to_numpy()
        #profile[np.isnan(profile[:,1]),1] = 0

        # Calculate FSG
        fsg_h1 = min_height
        fsg_h2 = min_height
        fsg_h1_temp = min_height
        fsg_h2_temp = min_height
        empty_counter = 0
        filled_counter = 0
        noise_bumps = 0
        has_gap = False

        for i in range(profile.shape[0]):
            if profile[i,1] < fsg_threshold:
                # Bin is "empty"
                # Decide whether to reset filled_counter
                if filled_counter != 0:
                    noise_bumps += 1
                    if noise_bumps > noise_bumps_allowed:
                        filled_counter = 0

                # If empty_counter is zero, set height as bottom of fsg. Start counting from here.
                if empty_counter == 0:
                    fsg_h1_temp = profile[i,0]
                    noise_bumps = 0

                # Increase empty counter
                empty_counter += 1

                # confirm if a gap exists
                if empty_counter >= min_contiguous_filled / bin_height:
                    has_gap = True

                # If empty space meets criteria, save the height we've been counting from. Ensure fsg_h1 is only set once.
                if (fsg_h1==min_height) and (empty_counter >= (min_contiguous_empty / bin_height)):
                    fsg_h1 = fsg_h1_temp
                    filled_counter = 0 # Record fsg top at next "filled" bin

            elif profile[i,1] >= fsg_threshold:
                # Bin is "filled"
                # Decide whether to reset empty_counter
                if empty_counter != 0:
                    noise_bumps += 1
                    if noise_bumps > noise_bumps_allowed:
                        empty_counter = 0

                # If filled_counter is zero, set height as top of fsg. Start counting from here.
                if filled_counter==0:
                    fsg_h2_temp = profile[i,0]
                    noise_bumps = 0

                # Increase counter
                filled_counter += 1

                # If filled space meets criteria, save height and we're done
                if (filled_counter >= min_contiguous_filled / bin_height) and has_gap:
                    fsg_h2 = fsg_h2_temp
                    break

            else:
                # Data is probably na
                continue
        if fsg_h1==min_height:
            fsg_h1 = 0
        if fsg_h2==min_height:
            fsg_h2 = 0
        summary.loc[summary[plot_id_col] == plot_id, 'fsg_h1'] = fsg_h1
        summary.loc[summary[plot_id_col] == plot_id, 'fsg_h2'] = fsg_h2
        summary.loc[summary[plot_id_col] == plot_id, 'fsg'] = fsg_h2 - fsg_h1

        # Calculate effective canopy bulk density
        cbd_window_size_bins = int(3.96 / bin_height) # 3.96m is the 13-foot running mean used in FVS-FFE
        cbd = profile[:,1]
        cbd_max = float(smooth_w_running_avg(cbd, cbd_window_size_bins).max())
        summary.loc[summary[plot_id_col] == plot_id, 'cbd_max'] = cbd_max

        # Calculate canopy height and ratio
        ch = profile[profile[:, 1] > 0][:, 0].max()  # Max height where val>0
        cr = (ch - fsg_h2) / ch
        summary.loc[summary[plot_id_col] == plot_id, 'canopy_height'] = ch
        summary.loc[summary[plot_id_col] == plot_id, 'canopy_ratio'] = cr

    return summary



def calculate_species_proportions(profile_data: pd.DataFrame,
                                  species_cols: list[str],
                                  result_col: str)->pd.DataFrame:
    """
    Convert CBD values to proportion of total CBD.

    Args:
        profile_data:
        species_cols:
        result_col:

    Returns:

    """

    profile_data = profile_data.copy()

    # Get species distributions
    profile_data[result_col] = profile_data[species_cols].sum(axis=1)
    for col in species_cols:
        # Get species composition percentage for each height bin
        profile_data[col] = (profile_data[col] / profile_data[result_col])
        # Forward fill percentages for nan or inf bins resulting from divide by zero error
        profile_data.loc[~np.isfinite(profile_data[col]), col] = np.nan
        profile_data[col] = profile_data[col].ffill()

    return profile_data



def smooth(values, smoothing_factor):
    if smoothing_factor > 0:
        return smooth_w_spline(values,smoothing_factor)
    else:
        return np.array(values)


def smooth_w_running_avg(values,window_size=40):
    import numpy as np
    from numba import njit,float64,int32,prange

    values = np.array(values,dtype=np.float64)
    values = np.pad(values, window_size, 'constant', constant_values=0)
    window_size = np.int32(window_size)

    @njit(float64[:](float64[:], int32), parallel=True)
    def running_average(values, window_size):
        n = values.shape[0]
        result = np.empty(n, dtype=np.float64)
        half_window = window_size // 2

        for i in prange(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            result[i] = np.nanmean(values[start:end])

        return result

    values_smooth = running_average(values, window_size)
    values_smooth = values_smooth[window_size:-window_size]
    return values_smooth

def smooth_w_spline(values,sigma=.01):
    import numpy as np
    from scipy import interpolate

    values = np.array(values)

    if len(values)>=5:
        x = np.arange(values.shape[0])  # Independent variable

        # Create a smoothing spline
        spline = interpolate.UnivariateSpline(x, values, s=sigma)

        # Smooth the data
        values_smooth = spline(x)
    else:
        values_smooth = values

    return values_smooth


def interp2D_w_nearest_neighbor_extrapolation(xy_train, values_train, xy_predict):
    import scipy

    f = scipy.interpolate.LinearNDInterpolator(xy_train, values_train)
    # evaluate the original interpolator. Out-of-bounds values are nan.
    values_predict = f(xy_predict)
    nans = np.isnan(values_predict)
    if nans.any():
        # Build a KD-tree for efficient nearest neighbor search.
        tree = scipy.spatial.cKDTree(xy_train)
        # Find the nearest neighbors for the NaN points.
        distances, indices = tree.query(xy_predict[nans], k=1)
        # Replace NaN values with the values from the nearest neighbors.
        values_predict[nans] = values_train[indices]
    return values_predict


def interp2D_w_cubic_extrapolation(xy_train, values_train, xy_predict):
    import scipy

    f = scipy.interpolate.LinearNDInterpolator(xy_train, values_train)
    # evaluate the original interpolator. Out-of-bounds values are nan.
    values_predict = f(xy_predict)
    nans = np.isnan(values_predict)
    if nans.any():
        import statsmodels.api as sm
        def create_dmatrix(xy):
            dmatrix = np.ones([xy.shape[0],7],dtype=float)
            dmatrix[:,1] = xy[:,0]
            dmatrix[:, 2] = xy[:,1]
            dmatrix[:, 3] = xy[:, 0] ** 2
            dmatrix[:, 4] = xy[:, 0] ** 3
            dmatrix[:, 5] = xy[:, 1] ** 2
            dmatrix[:, 6] = xy[:, 1] ** 3
            return dmatrix
        model = sm.OLS(values_train,create_dmatrix(xy_train)).fit()
        values_predict[nans] = model.predict(create_dmatrix(xy_predict[nans]))
    return values_predict

def interpolate_flightpath(points, flightpath):
    from scipy.interpolate import interp1d
    interpolator_x = interp1d(flightpath['GpsTime'].to_numpy(),flightpath['X'].to_numpy())
    interpolator_y = interp1d(flightpath['GpsTime'].to_numpy(), flightpath['Y'].to_numpy())
    interpolator_z = interp1d(flightpath['GpsTime'].to_numpy(), flightpath['Z'].to_numpy())
    x = interpolator_x(points['GpsTime'])
    y = interpolator_y(points['GpsTime'])
    z = interpolator_z(points['GpsTime'])
    return np.stack([x,y,z],1)

def _default_folder_setup(export_folder,
                          pad_dir=True,
                          dem_dir=True,
                          points_dir=True,
                          pad_summary_dir=True,
                          plot_summary_dir=True):
    from pathlib import Path
    import os

    if not Path(export_folder).exists():
        os.makedirs(export_folder)
    if dem_dir and not Path(export_folder).joinpath('DEM').exists():
        os.mkdir(Path(export_folder).joinpath('DEM'))
    if points_dir and not Path(export_folder).joinpath('Points').exists():
        os.mkdir(Path(export_folder).joinpath('Points'))
    if pad_dir and not Path(export_folder).joinpath('PAD').exists():
        os.mkdir(Path(export_folder).joinpath('PAD'))
    if pad_summary_dir and not Path(export_folder).joinpath('PAD_Summary').exists():
        os.mkdir(Path(export_folder).joinpath('PAD_Summary'))
    if plot_summary_dir and not Path(export_folder).joinpath('Plot_Summary').exists():
        os.mkdir(Path(export_folder).joinpath('Plot_Summary'))

def _default_postprocessing(grid, plot_name,
                            export_folder,
                            plot_radius=11.3,
                            max_occlusion=.8,
                            sigma1=.1,
                            fill_occlusion=False,
                            min_pad_foliage=.01,
                            max_pad_foliage=6,
                            export_grid=True,
                            export_dem=True,
                            export_pad_summary=True,
                            export_plot_summary=True
                            )->Tuple['pd.DataFrame','pd.DataFrame']:
    import os
    import pandas as pd
    if fill_occlusion:
        grid.interpolate_occlusion_idw(max_occlusion=max_occlusion, max_pad_foliage=max_pad_foliage, k=8)
        max_occlusion=999
    grid.filter_pad_noise_ivf()
    grid.gaussian_filter_PAD(sigma=sigma1)
    grid.classify_foliage_with_PAD(max_occlusion=max_occlusion, min_pad_foliage=min_pad_foliage, max_pad_foliage=max_pad_foliage)
    profile = grid.summarize_by_height(clip_radius=plot_radius)
    summary = grid.calculate_dem_metrics(clip_radius=plot_radius)
    summary['canopy_cover'] = grid.calculate_canopy_cover(clip_radius=plot_radius)
    summary['plot_id'] = plot_name
    summary = pd.DataFrame(summary, index=[0])
    if export_grid:
        grid.export_grid_as_csv(os.path.join(export_folder, 'PAD/', plot_name) + '.csv')
    if export_dem:
        grid.export_dem_as_csv(os.path.join(export_folder, 'DEM/', plot_name) + '.csv')
    if export_pad_summary:
        profile.write_csv(os.path.join(export_folder, 'PAD_Summary/', plot_name) + '.csv')
    if export_plot_summary:
        summary.to_csv(os.path.join(export_folder, 'Plot_Summary/', plot_name) + '.csv', index=False)
    profile = profile.to_pandas()
    return profile,summary

def estimate_foliage_from_treelist(treelist:pd.DataFrame,
                                   lma_ref_spcd:pd.DataFrame=None,
                                   lma_ref_spgrpcd:pd.DataFrame=None,
                                   division="",
                                   dia_col = "DIA",
                                   ht_col = "HT",
                                   tpa_col = "TPA_UNADJ",
                                   species_col = "SPCD",
                                   species_group_col = "SPGRPCD",
                                   foliage_mass_col = "DRYBIO_FOLIAGE",
                                   input_metric=True,
                                   output_metric=True):
    """Make canopy bulk density profile based on tree list and allometric equations from NSVB.

    Args:
        treelist (pd.DataFrame): table containing tree data, e.g. in FIA format
        lma_ref_spcd (pd.DataFrame): table containing leaf mass per area values, organized by species code.
            Must have column "LMA" with units kg/m^2.
        lma_ref_spgrpcd (pd.DataFrame): table containing leaf mass per area values, organized by species group code.
            Must have column "LMA" with units kg/m^2.
    lma_ref_spcd: LMA reference SPCD file

    Input treelist must be in FIA format. lma_ref

    For imperial units:
        dia_col: in
        ht_col: ft
        tpa_col: trees per acre
        foliage_mass_col: lb
        leaf_area_col: ft^2

    For metric units:
        dia_col: cm
        ht_col: m
        tpa_col: trees per hectare
        foliage_mass_col: kg
        leaf_area_col: m^2

    Units in other columns are not considered"""

    from nsvb import estimators

    if input_metric:
        treelist[dia_col] /= 2.54
        treelist[ht_col] *= 3.2808
        treelist[tpa_col] *= 2.471

    treelist[foliage_mass_col] = 0.
    for i in treelist.index:
        treelist.loc[i,foliage_mass_col] = estimators.total_foliage_dry_weight(treelist.loc[i,species_col],
                                                                               treelist.loc[i,dia_col],
                                                                               treelist.loc[i,ht_col],
                                                                               division)
    if lma_ref_spcd is not None:
        if 'SPCD_OG' in treelist.columns:
            treelist = treelist.merge(lma_ref_spcd[[species_col, 'LMA']], left_on='SPCD_OG',right_on=species_col,suffixes=['','_y'], how='left')
        else:
            treelist = treelist.merge(lma_ref_spcd[[species_col, 'LMA']], on=species_col, how='left')
        treelist['LMA_SPCD'] = treelist['LMA']
        if lma_ref_spgrpcd is not None:
            treelist = treelist.drop(columns='LMA')

    if lma_ref_spgrpcd is not None:
        treelist = treelist.merge(lma_ref_spgrpcd[[species_group_col,'LMA']], on=species_group_col, how='left')
        treelist['LMA_SPGRPCD'] = treelist['LMA']
        if lma_ref_spcd is not None:
            treelist['LMA'] = treelist['LMA_SPCD']
            treelist.loc[treelist['LMA'].isna(),'LMA'] = treelist.loc[treelist['LMA'].isna(),'LMA_SPGRPCD']

    # Convert lb to kg for leaf area calculation
    treelist[foliage_mass_col] /= 2.2046

    # Calculate leaf area
    if 'LMA' in treelist.columns:
        treelist['LEAF_AREA'] = treelist[foliage_mass_col] / treelist['LMA']
        # Convert metric to sq feet
        if not output_metric:
            treelist['LEAF_AREA'] *= 10.7639

    # Convert in,ft,tpa to cm,m,tpha
    if output_metric:
        treelist[dia_col] *= 2.54
        treelist[ht_col] /= 3.2808
        treelist[tpa_col] /= 2.471

    # Convert kg to lb
    if not output_metric:
        treelist[foliage_mass_col] *= 2.2046

    return treelist


def profiles_from_treelist(treelist:pd.DataFrame,
                           plot_name_col = "PLT_CN",
                           dia_col = "DIA",
                           ht_col = "HT",
                           crown_ratio_col = "CR",
                           tpa_col = "TPA_UNADJ",
                           species_col = "SPCD",
                           species_group_col = "SPGRPCD",
                           foliage_mass_col = "DRYBIO_FOLIAGE",
                           ht_interval:float=0.2,
                           area_factor:float=10000,
                           by_species:bool=True):
    """Make canopy bulk density and/or leaf area density profiles from treelist.

    Treelist must be in FIA format. This process is units-agnostic except for area factor, which represents area2/area1
    where input units are trees/area1 and output units are kg/(ht_unit*area2). For input trees/ha and output kg/m^3, area
    factor is 10000 m^2/ha."""

    tree_bins = []
    for i in treelist.index:
        ht = treelist.loc[i,ht_col]
        cr = treelist.loc[i,crown_ratio_col]
        if np.isnan(ht) or np.isnan(cr):
            raise ValueError('HT and CR must not be nan')

        max_ht_bin = int(round(ht / ht_interval))
        min_ht_bin = int(round((ht * (1 - cr / 100)) / ht_interval))
        bins = np.arange(min_ht_bin,max_ht_bin+1)
        df = pd.DataFrame({plot_name_col:treelist.loc[i,plot_name_col],'HT_BIN':bins, ht_col: bins*ht_interval,
                           species_col:treelist.loc[i,species_col]})
        if foliage_mass_col in treelist.columns:
            df['CBD']=treelist.loc[i,foliage_mass_col] * treelist.loc[i,tpa_col] / area_factor / len(bins)
        if 'LEAF_AREA' in treelist.columns:
            df['LAD']=treelist.loc[i,'LEAF_AREA'] * treelist.loc[i,tpa_col] / area_factor / len(bins)
        if 'LMA' in treelist.columns:
            df['LMA']=treelist.loc[i,'LMA']
        tree_bins.append(df)
    tree_bins = pd.concat(tree_bins)

    if by_species:
        profiles = tree_bins.pivot_table(index=[plot_name_col, 'HT_BIN', species_col],
                                         aggfunc={'CBD':'sum', 'LAD':'sum', 'LMA':'mean'}).reset_index()
    else:
        tree_bins = tree_bins.drop(columns=species_col)
        tree_bins['LMAxCBD'] = tree_bins['LMA'] * tree_bins['CBD']
        profiles = tree_bins.pivot_table(index=[plot_name_col, 'HT_BIN'],
                                         aggfunc={'CBD': 'sum', 'LAD': 'sum', 'LMAxCBD': 'sum'}).reset_index()
        profiles['LMA'] = profiles['LMAxCBD'] / profiles['CBD']
        profiles = profiles.drop(columns='LMAxCBD')

    profiles.insert(2, ht_col, profiles['HT_BIN'] * ht_interval)
    return profiles


