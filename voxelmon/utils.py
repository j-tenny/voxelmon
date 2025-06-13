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

def open_file_pdal(filepath,bounds=None,calculate_height=True,reproject_to=None)->Tuple['pl.DataFrame',str]:
    """Read a file to a polars dataframe with pdal. Returns pl.DataFrame and crs

    Args:
        filepath (str): Path to ALS file readable by pdal. Type is inferred by extension.
        bounds (str): Clip extents of the resource in 2 or 3 dimensions, formatted as pdal-compatible string,
            e.g.: ([xmin, xmax], [ymin, ymax], [zmin, zmax]). If omitted, the entire dataset will be selected.
            The bounds can be followed by a slash (‘/’) and a spatial reference specification to apply to the bounds.
        calculate_height (bool): Calculate height above ground for each point using Delauney triangulation
        reproject_to (str): Reproject to this CRS. Use format 'EPSG:5070' or PROJ. If None, no reprojection will be done.
                """
    import pdal
    import polars as pl

    result=0

    filters = []
    if bounds is not None:
        filters.append(pdal.Reader(filepath, bounds=bounds))
    else:
        filters.append(pdal.Reader(filepath))

    if reproject_to is not None:
        filters.append(pdal.Filter.reprojection(out_srs=reproject_to))

    if calculate_height:
        count = 10
        while result == 0 and count < 100:
            filters_temp = filters + [pdal.Filter.hag_delaunay(count=count)]
            try:
                pipeline = pdal.Pipeline(filters_temp)
                result = pipeline.execute()
            except:
                count *= 2
    else:
        pipeline = pdal.Pipeline(filters)
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


def summarize_profiles(profiles, bin_height=.1, min_height=1., fsg_threshold=.011, cbd_col='cbd_pred',
                       height_col='height', pad_col='pad', occlusion_col='occluded', plot_id_col='Plot_ID'):
    profiles_filter_h = profiles[profiles[height_col] >= min_height]
    summary = profiles_filter_h.pivot_table(index=plot_id_col,
                                            aggfunc={cbd_col: 'sum', pad_col: 'sum', occlusion_col: 'mean'}
                                            ).reset_index()

    summary.columns = [plot_id_col, 'cfl', 'occlusion', 'plant_area']
    summary['cfl'] *= bin_height
    summary['plant_area'] *= bin_height
    summary['fsg'] = 0.
    summary['fsg_h1'] = float(min_height)
    summary['fsg_h2'] = float(min_height)
    summary['canopy_height'] = 0.
    summary['canopy_ratio'] = 0.
    summary.insert(1, 'cbd_max', 0.)

    cbd_window_size_bins = int(3.96 / bin_height)  # 3.96m is the 13-foot running mean used in FVS-FFE

    for plot_id in profiles[plot_id_col].unique():
        profile = profiles[profiles[plot_id_col] == plot_id]
        profile = profile.sort_values(by=height_col)
        profile = profile[[height_col, cbd_col]].to_numpy()
        heights = np.arange(profile[:, 0].min(), profile[:, 0].max() + bin_height, bin_height)
        cbd = np.interp(heights, profile[:, 0], profile[:, 1])
        cbd_smooth = smooth_w_running_avg(cbd, cbd_window_size_bins)

        # Calculate FSG
        if cbd_smooth[0] < fsg_threshold:
            # If profile starts "empty", get lowest height where profile is "filled"
            fsg_h1 = 0
            filled_mask = cbd_smooth >= fsg_threshold
            if filled_mask.sum() > 0:
                fsg_h2 = np.min(heights[filled_mask])
            else:
                fsg_h2 = 0
        else:
            # If profile starts "filled", get lowest height where profile is "empty"
            fsg_h1 = np.min(heights[cbd_smooth < fsg_threshold])
            # Remove values below fsg_h1
            heights_clip = heights[heights >= fsg_h1]
            cbd_clip = cbd[heights >= fsg_h1]
            # Get lowest height where profile is "filled"
            filled_mask = cbd_clip >= fsg_threshold
            if filled_mask.sum() > 0:
                fsg_h2 = np.min(heights_clip[filled_mask])
            else:
                fsg_h2 = fsg_h1

        summary.loc[summary[plot_id_col] == plot_id, 'fsg_h1'] = fsg_h1
        summary.loc[summary[plot_id_col] == plot_id, 'fsg_h2'] = fsg_h2
        summary.loc[summary[plot_id_col] == plot_id, 'fsg'] = fsg_h2 - fsg_h1

        # Calculate effective canopy bulk density
        cbd_max = cbd_smooth.max()
        summary.loc[summary[plot_id_col] == plot_id, 'cbd_max'] = cbd_max

        # Calculate canopy height and ratio
        ch = profile[profile[:, 1] > 0][:, 0].max()  # Max height where val>0
        cr = (ch - fsg_h2) / ch
        summary.loc[summary[plot_id_col] == plot_id, 'canopy_height'] = ch
        summary.loc[summary[plot_id_col] == plot_id, 'canopy_ratio'] = cr

    return summary


def summarize_profiles_from_grid(cbd_arr, z_coords, interpolated_resolution=.1, fsg_threshold=.011):
    from numba import njit, float64,jit,prange
    import matplotlib.pyplot as plt
    heights = np.arange(z_coords.min(), z_coords.max() + interpolated_resolution, interpolated_resolution)
    cbd_max_arr = np.zeros([cbd_arr.shape[0], cbd_arr.shape[1]], float)
    cfl_arr = np.zeros_like(cbd_max_arr)
    fsg_h1_arr = np.zeros_like(cbd_max_arr)
    fsg_h2_arr = np.zeros_like(cbd_max_arr)
    fsg_arr = np.zeros_like(cbd_max_arr)
    ch_arr = np.zeros_like(cbd_max_arr)
    cr_arr = np.zeros_like(cbd_max_arr)

    @njit((float64[:, :, :],
           float64[:],
           float64[:],
           float64,
           float64[:, :],
           float64[:, :],
           float64[:, :],
           float64[:, :],
           float64[:, :],
           float64[:, :],
           float64[:, :]), parallel=True)
    def summarize_grid(cbd_arr,
                       heights_og,
                       heights,
                       fsg_threshold,
                       cbd_max_arr,
                       cfl_arr,
                       fsg_h1_arr,
                       fsg_h2_arr,
                       fsg_arr,
                       ch_arr,
                       cr_arr):

        for i in prange(cbd_arr.shape[0]):
            for j in range(cbd_arr.shape[1]):

                cbd_og = cbd_arr[i, j, :]

                # Interpolate cbd from heights_og to heights
                cbd = np.zeros_like(heights)

                k = 0
                if heights_og.size > 1:
                    h0 = heights_og[0]
                    h1 = heights_og[1]
                    cbd0 = cbd_og[0]
                    cbd1 = cbd_og[1]
                    slope = (cbd1 - cbd0) / (h1 - h0)

                    for ht_idx in range(heights.size):
                        ht = heights[ht_idx]

                        if ht <= heights_og[0]:
                            cbd[ht_idx] = cbd_og[0]
                        elif ht >= heights_og[-1]:
                            cbd[ht_idx] = cbd_og[-1]
                        else:
                            while ht >= heights_og[k + 1]:
                                k += 1
                                h0 = heights_og[k]
                                h1 = heights_og[k + 1]
                                cbd0 = cbd_og[k]
                                cbd1 = cbd_og[k + 1]
                                slope = (cbd1 - cbd0) / (h1 - h0)

                            cbd[ht_idx] = cbd0 + slope * (ht - h0)
                else:
                    cbd[0] = cbd_og[0]

                # Smooth with running average
                window_size = int(3.96 / interpolated_resolution)  # 3.96m is the 13-foot running mean used in FVS-FFE
                cbd_pad = np.full(cbd.size + 2 * window_size, 0.,
                                  dtype=np.float64)  # pad start and end of values with zeros
                cbd_pad[window_size:window_size + cbd.size] = np.copy(cbd)
                n = cbd_pad.size
                cbd_smooth_pad = np.empty(n, np.float64)
                half_window = window_size // 2
                for cbd_idx in range(n):
                    start = max(0, cbd_idx - half_window)
                    end = min(n, cbd_idx + half_window + 1)
                    cbd_smooth_pad[cbd_idx] = np.nanmean(cbd_pad[start:end])
                cbd_smooth = cbd_smooth_pad[window_size:-window_size]

                # Calculate FSG
                if cbd_smooth[0] < fsg_threshold:
                    # If profile starts "empty", get lowest height where profile is "filled"
                    fsg_h1 = 0
                    filled_mask = cbd_smooth >= fsg_threshold
                    if filled_mask.sum() > 0:
                        fsg_h2 = np.nanmin(heights[filled_mask])
                        ch = np.nanmax(heights[filled_mask])
                    else:
                        fsg_h2 = 0
                        ch = 0
                else:
                    # If profile starts "filled", get lowest height where profile is "empty"
                    empty_mask = cbd_smooth < fsg_threshold
                    if empty_mask.sum() > 0:
                        fsg_h1 = np.nanmin(heights[empty_mask])
                    else:
                        fsg_h1 = heights[-1]
                    # Remove values below fsg_h1
                    heights_clip = heights[heights >= fsg_h1]
                    cbd_clip = cbd[heights >= fsg_h1]
                    # Get lowest height where profile is "filled"
                    filled_mask = cbd_clip >= fsg_threshold
                    if filled_mask.sum() > 0:
                        fsg_h2 = np.nanmin(heights_clip[filled_mask])
                        ch = np.nanmax(heights_clip[filled_mask])
                    else:
                        fsg_h2 = fsg_h1
                        ch = fsg_h1

                fsg_h1_arr[i, j] = fsg_h1
                fsg_h2_arr[i, j] = fsg_h2
                fsg_arr[i, j] = fsg_h2 - fsg_h1
                ch_arr[i, j] = ch

                # Calculate effective canopy bulk density
                cbd_max = cbd_smooth.max()
                cbd_max_arr[i, j] = cbd_max

                # Calculate canopy fuel load
                if heights_og.size > 1:
                    bin_size_og = heights_og[1] - heights_og[0]
                else:
                    bin_size_og = heights_og[0]

                cfl_arr[i, j] = cbd_og.sum() * bin_size_og

                # Calculate canopy height and ratio
                if ch > 0:
                    cr = (ch - fsg_h2) / ch
                else:
                    cr = 1
                ch_arr[i, j] = ch
                cr_arr[i, j] = cr
                #
                # if cbd_max>0:
                #     if np.random.random() <= .005:
                #         plt.axhspan(fsg_h1, fsg_h2,alpha=0.5)
                #         plt.plot(cbd_smooth,heights,c='blue')
                #         plt.plot(cbd_og, heights_og,c='red', ls='--')
                #         plt.axvline(cbd_smooth.max(),c='black')
                #         plt.axvline(fsg_threshold,c='black',ls='--')
                #         plt.show()


    summarize_grid(cbd_arr,
                   z_coords,
                   heights,
                   fsg_threshold,
                   cbd_max_arr,
                   cfl_arr,
                   fsg_h1_arr,
                   fsg_h2_arr,
                   fsg_arr,
                   ch_arr,
                   cr_arr)

    return [cbd_max_arr, cfl_arr, fsg_h1_arr, fsg_h2_arr, fsg_arr, ch_arr, cr_arr]


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

def visualize_voxels(grid:'pl.DataFrame',
                     dem:'pl.DataFrame',
                     points:'pl.DataFrame'=None,
                     clip_extents='auto',
                     value_name:str='pad',
                     min_value_leaf:float=0.2,
                     max_value_leaf:float=6,
                     min_hag:float=0.1,
                     recenter_elev:bool=False)->None:
    """
    Show an interactive 3D viewer displaying voxel grid, DEM, and (optionally) point cloud.

    Args:

        grid (pl.DataFrame): Polars DataFrame containing voxel grid (one row for each voxel. Columns must
            include 'x', 'y', 'z', 'classification', 'hag' and value_name (default='pad')

        dem (pl.DataFrame): Polars DataFrame containing DEM in point format (row for each cell, columns xyz).

        clip_extents (str or list): Spatial extent to crop the visualization.
            Can be 'auto', a list of 4 floats [xmin, ymin, xmax, ymax], or 6 floats
            [xmin, ymin, zmin, xmax, ymax, zmax]. Defaults to 'auto'.

        value_name (str): Name of the voxel attribute used for scalar coloring.
            Defaults to 'pad'.

        min_value_leaf (float): Voxels with value below this threshold are considered "empty", not rendered.

        max_value_leaf (float): Voxels with value below this threshold are considered "leaves" and rendered along the
            yellow-green color scale. Voxels with value above this threshold are considered "non-foliage" and rendered in
            brown. Recommended value to visualize 'pad' from the BLK360 is 6. Recommended value to visualize 'pad' from
            ALS is 1.

        min_hag (float): Minimum height above ground for voxels to be rendered. Defaults to 0.1, consider 1.0 for ALS.

        recenter_elev (bool): If True, DEM and all data are centered around the DEM's mean elevation. Useful for
            ALS data. Defaults to False.

    """
    import numpy as np
    import pyvista as pv
    import polars as pl

    grid.columns = [col.lower() for col in grid.columns]
    grid = grid.select(['x','y','z','hag','classification',value_name])
    cell_size = round(grid[1, 0] - grid[0, 0], 5)

    # Determine extents
    if clip_extents == 'auto':
        extents = np.concatenate([
            grid[:, 0:3].min().to_numpy().flatten(),
            grid[:, 0:3].max().to_numpy().flatten()
        ])
    elif len(clip_extents) == 4:
        extents = np.concatenate([
            grid[:, 0:3].min().to_numpy().flatten(),
            grid[:, 0:3].max().to_numpy().flatten()
        ])
        extents[0:2] = clip_extents[0:2]
        extents[3:5] = clip_extents[2:4]
    elif len(clip_extents) == 6:
        extents = np.array(clip_extents)
    else:
        raise ValueError("clip_extents must be 'auto', or a list of 4 or 6 floats")

    # Clip grid
    grid = grid.filter(
        (grid[:, 0] >= extents[0]) & (grid[:, 0] <= extents[3]) &
        (grid[:, 1] >= extents[1]) & (grid[:, 1] <= extents[4]) &
        (grid[:, 2] >= extents[2]) & (grid[:, 2] <= extents[5])
    )

    # Load and clip DEM
    dem.columns = [col.lower() for col in dem.columns]
    dem = dem.select(['x','y','z'])
    dem = dem.filter(
        (dem[:, 0] >= extents[0]) & (dem[:, 0] <= extents[3]) &
        (dem[:, 1] >= extents[1]) & (dem[:, 1] <= extents[4])
    )
    dem = dem.to_numpy()

    # Recenter elevation coordinates to zero
    if recenter_elev:
        center = dem.mean(axis=0)
        dem -= center
        grid = grid.with_columns([
            pl.col('x') - center[0],
            pl.col('y') - center[1],
            pl.col('z') - center[2]
        ])
    else:
        center = None

    if points is not None:
        show_points=True
    else:
        show_points=False

    if show_points:
        points.columns = [col.lower() for col in points.columns]
        points = points.select(['x','y','z'])
        points = points.filter(
            (points[:, 0] >= extents[0]) & (points[:, 0] <= extents[3]) &
            (points[:, 1] >= extents[1]) & (points[:, 1] <= extents[4]) &
            (points[:, 2] >= extents[2]) & (points[:, 2] <= extents[5])
            )
        points = points.to_numpy()

        if recenter_elev:
            points -= center
        pts_all = pv.PolyData(points[:, :3])

    shape = ((dem.max(0) - dem.min(0)) / cell_size).round().astype(int) + 1
    dem_grid = pv.StructuredGrid(
        dem[:, 0].reshape(shape[0:2]),
        dem[:, 1].reshape(shape[0:2]),
        dem[:, 2].reshape(shape[0:2])
    ).texture_map_to_plane()

    # Classification masks
    leaf_mask = ((pl.col('classification') > 0) &
                 (pl.col('hag') >= min_hag) &
                 (pl.col(value_name) >= min_value_leaf) &
                 (pl.col(value_name) <= max_value_leaf))
    wood_mask = ((pl.col('classification') > 0) &
                 (pl.col('hag') >= min_hag) &
                 (pl.col(value_name) > max_value_leaf)
                 )
    occluded_mask = (pl.col('classification') == -1)

    base_cube = pv.Cube(center=(0, 0, 0), x_length=cell_size, y_length=cell_size, z_length=cell_size)
    filled_pts = pv.PolyData(grid.filter(leaf_mask)[:, :3].to_numpy())
    filled_values = grid.filter(leaf_mask)[value_name].to_numpy()
    filled_values = np.repeat(filled_values, 6)
    filled_glyphs = filled_pts.glyph(orient=False, scale=False, geom=base_cube)

    plotter = pv.Plotter(lighting='three lights')
    plotter.set_background('black', top='white')
    plotter.enable_ssao(radius=1)
    plotter.enable_terrain_style(mouse_wheel_zooms=True, shift_pans=True)
    plotter.add_axes(interactive=True)
    pv.global_theme.full_screen = True

    scalar_bar_args = {
        'title': 'Plant Area Density (m²/m³)',
        'label_font_size': 24,
        'title_font_size': 24,
        'color': 'white'
    }

    # Add filled voxels
    filled_actor = plotter.add_mesh(
        filled_glyphs, scalars=filled_values, clim=[0, max_value_leaf],
        opacity=1, cmap='YlGn', show_scalar_bar=True, scalar_bar_args=scalar_bar_args
    )

    # Add wood voxels
    if len(grid.filter(wood_mask)) > 0:
        wood_pts = pv.PolyData(grid.filter(wood_mask)[:, :3].to_numpy())
        wood_glyphs = wood_pts.glyph(orient=False, scale=False, geom=base_cube)
        wood_actor = plotter.add_mesh(wood_glyphs, color='brown', opacity=1)
    else:
        wood_actor = None

    # Add occluded voxels
    if len(grid.filter(occluded_mask)) > 0:
        occ_pts = pv.PolyData(grid.filter(occluded_mask)[:, :3].to_numpy())
        occ_glyphs = occ_pts.glyph(orient=False, scale=False, geom=base_cube)
        occ_actor = plotter.add_mesh(occ_glyphs, color=True, opacity=0.25)
    else:
        occ_actor = None

    # Add scanner and DEM
    plotter.add_mesh(pv.Cylinder((0, 0, -1), direction=(0, 0, 1), radius=0.05, height=2), color='black')
    plotter.add_mesh(pv.Sphere(radius=0.2), color='black')
    plotter.add_mesh(dem_grid, color='#8A5D36', smooth_shading=True)

    if show_points:
        pts_actor = plotter.add_mesh(pts_all, scalars=points[:, 2], clim=[-3, 12], show_scalar_bar=False)
        pts_actor.SetVisibility(False)

    # Slider to adjust filled voxel opacity
    def set_filled_opacity(value):
        filled_actor.GetProperty().SetOpacity(value)
        if wood_actor:
            wood_actor.GetProperty().SetOpacity(value)

    plotter.add_slider_widget(set_filled_opacity, [0, 1], title='Filled Voxel Opacity',
                              pointa=(.1, .9), pointb=(.3, .9), value=1)

    # Slider to adjust occluded voxel opacity
    def set_occluded_opacity(value):
        if occ_actor:
            occ_actor.GetProperty().SetOpacity(value)

    plotter.add_slider_widget(set_occluded_opacity, [0, 1], title='Occluded Voxel Opacity',
                              pointa=(.1, .7), pointb=(.3, .7), value=0.25)

    # Checkbox to show/hide points
    if show_points:
        plotter.add_checkbox_button_widget(lambda flag: pts_actor.SetVisibility(flag), value=False, size=20)

    plotter.show()

