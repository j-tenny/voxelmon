import numpy as np
import voxelmon
import pdal_piper
import rasterio
import polars as pl
import pdal
import pdal_piper.stages as ps





def bin2D(points_df, function, cell_size, origin=(0, 0)) -> 'polars.DataFrame':
    """Aggregate point cloud to a 2D grid and apply a polars function

    points_df: a polars dataframe with columns 'X', 'Y', 'Z'
    function: a function compatible with polars.DataFrame.aggregate(); may need to specify col name e.g. pl.min('z')
    cell_size: float value for output raster resolution
    origin: origin of grid relative to coordinates
    """
    # Function should be from polars and

    import polars as pl
    import numpy as np

    # Get function value for each bin
    points_df = points_df.with_columns(pl.col('X').sub(origin[0]).floordiv(cell_size).cast(pl.Int32).alias('XBin'),
                                       pl.col('Y').sub(origin[1]).floordiv(cell_size).cast(pl.Int32).alias('YBin'))

    bin_vals = points_df.group_by(['XBin', 'YBin']).agg(function)

    # Get df containing all possible bins
    binminx = points_df['XBin'].min()
    binminy = points_df['YBin'].min()
    binmaxx = points_df['XBin'].max()
    binmaxy = points_df['YBin'].max()

    ybins, xbins = np.meshgrid(np.arange(binminy, binmaxy + 1),
                               np.arange(binminx, binmaxx + 1),
                               indexing='ij')

    bins_df = pl.DataFrame({'YBin': ybins.flatten().astype(np.int32),
                            'XBin': xbins.flatten().astype(np.int32)})

    return bins_df.join(bin_vals, ['YBin', 'XBin'], 'left')




def read_pdal(path,bounds):
    import pdal
    import polars as pl
    pipeline = pdal.Pipeline([pdal.Reader(path,bounds=bounds)])
    pipeline.execute()
    return pl.DataFrame(pipeline.arrays[0]), pipeline.srswkt2

window_size = 10
url = pdal_piper.USGS_3dep_Finder([782848.5,4213339.2,787769.2,4217510.1],26912).select_url(0)
tiler = pdal_piper.Tiler([782848.5,4213339.2,787769.2,4217510.1],tile_size=1000,crs='EPSG:26912',convert_units=False)
tiles = tiler.get_tiles(format_as_pdal_str=True,flatten=True)
arr, crs = read_pdal(url,tiles[0])
i = 0
for tile in tiles:
    als = voxelmon.ALS(url,tile)
    lad = als.simple_pad(min_height=1.5, bin_size_xy=window_size)
    lad = lad.filter(pl.col('Z') <= 4)
    lad = lad.with_columns(pl.when(pl.col('LAD')<0).then(None).otherwise('LAD'))
    lad_grid = bin2D(lad,pl.mean('LAD'),window_size )

    # Convert to raster
    nx = lad_grid['XBin'].n_unique()
    ny = lad_grid['YBin'].n_unique()
    grid = lad_grid.sort(['YBin', 'XBin'], descending=[True, False])['LAD'].to_numpy().reshape([ny, nx])

    # Get coordinates of upper left corner
    ul_x = lad_grid['XBin'].min() * window_size
    ul_y = lad_grid['YBin'].max() * window_size

    transform = rasterio.transform.from_origin(ul_x, ul_y, window_size, window_size)

    metadata = {
        'driver': 'GTiff',
        'dtype': rasterio.float32,
        'nodata': None,
        'width': nx,
        'height': ny,
        'count': 1,  # Number of bands
        'crs': crs,  # Coordinate reference system
        'transform': transform,
    }

    with rasterio.open(f'D:/DataWork/ALS_ridgeway/ridgeway_{i}.tif', 'w', **metadata) as dst:
        dst.write(grid.astype(rasterio.float32), 1)  # Write to band 1

    i+=1
