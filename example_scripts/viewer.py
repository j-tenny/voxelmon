# View processed plant area density grid in 3D
# Requires pyvista `pip install pyvista`

from pathlib import Path
import polars as pl
import voxelmon

grid_path = 'G:/wscatclover/PAD/wscatclover00041024.csv'
clip_extents = 'auto'

grid_path=Path(grid_path)
grid = pl.read_csv(grid_path)
basename = grid_path.name
dem_path = grid_path.parents[1] / 'DEM' / basename
dem = pl.read_csv(dem_path)

points_path = grid_path.parents[1] / 'Points' / basename
if points_path.exists():
    points = pl.read_csv(points_path)
else:
    points = None

voxelmon.visualize_voxels(grid,
                          dem,
                          points,
                          clip_extents='auto',
                          value_name='PAD',
                          min_value_leaf=.01,
                          max_value_leaf=6,
                          min_hag=.1,
                          recenter_elev=False)