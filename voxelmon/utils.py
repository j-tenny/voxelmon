def get_files_list(directory, keyword):
    import os
    import fnmatch
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, f'*{keyword}*'):
            matching_files.append(os.path.join(root, filename))
    return matching_files


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
        result


def bin3D(pulses, function, cellSize,asArray = True, binExtents=None):
    # Function should be from polars and should specify a column name x, y, or z, e.g. pl.min('z')
    # binExtents must be iterable formatted as [xBinMin,yBinMin,zBinMin,xBinMax,yBinMax,zBinMax] noninclusive of end
    # if binExtents is None, extents are automatically pulled from pulses extents, may need to clip first
    import polars as pl
    import numpy as np
    try:
        points_df = pl.DataFrame({'x': pulses.xyz[:, 0], 'y': pulses.xyz[:, 1], 'z': pulses.xyz[:, 2]})
    except:
        points_df = pl.DataFrame({'x': pulses[:, 0], 'y': pulses[:, 1], 'z': pulses[:, 2]})

    points_df = points_df.with_columns(pl.col('x').floordiv(cellSize).cast(pl.Int32).alias('xBin'),
                                       pl.col('y').floordiv(cellSize).cast(pl.Int32).alias('yBin'),
                                       pl.col('z').floordiv(cellSize).cast(pl.Int32).alias('zBin'))

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