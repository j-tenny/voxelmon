import numpy as np
import polars as pl
import pandas as pd
import pandas as pd
from numba import jit,njit,guvectorize,prange,float32,void,uint16,int64,uint32,int32,float64
from typing import Union, Sequence

import voxelmon


class Grid:
    """Class to compute and store gridded values including voxelized PAD and terrain elevation

    Grid values are stored as 3D arrays with dimensions x, then y, then z
    """
    def __init__(self, extents:Sequence[float], cell_size:float):
        """Initialize Grid

        Args:
            extents (Sequence[float]): Extents of the grid formatted as [xmin, ymin, zmin, xmax, ymax, zmax]
            cell_size (float): cell size
        """
        self.cell_size = np.float64(cell_size)
        self._set_extents(extents)

        x_centers = np.arange(self.extents[0], self.extents[3], cell_size) + cell_size / 2
        y_centers = np.arange(self.extents[1], self.extents[4], cell_size) + cell_size / 2
        z_centers = np.arange(self.extents[2], self.extents[5], cell_size) + cell_size / 2

        self.shape = [len(x_centers), len(y_centers), len(z_centers)]

        z, y, x = np.meshgrid(z_centers, y_centers, x_centers, indexing='ij')

        self.centers = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T  # Grid centers increment along x, then y, then z
        self.centers_xy = self.centers[:self.shape[0] * self.shape[1], :2]

        self.p_directed = np.zeros(self.shape, dtype=np.float64)
        self.p_transmitted = np.zeros(self.shape, dtype=np.float64)
        self.p_intercepted = np.zeros(self.shape, dtype=np.float64)
        self.classification = np.zeros(self.shape, dtype=np.int8)

        self.classification_key = {}

        self.geometric_features = np.zeros([self.centers.shape[0], 4], np.float32) - 50


    @classmethod
    def from_dims(cls, cell_size:float, grid_half_length_x:float, grid_half_length_y:float, grid_height:float, grid_bottom:float=None)->'Grid':
        """Define grid by its dimensions

        Args:
            cell_size (float): cell size
            grid_half_length_x (float): half the width of the grid in x dim (i.e. plot radius)
            grid_half_length_y (float): half the width of the grid in y dim (i.e. plot radius)
            grid_height (float): height of the grid above origin (height of tallest tree)
            grid_bottom (float): distance from origin to bottom of the grid (zdist from origin to lowest terrain point)
            """
        if grid_bottom is None:
            grid_extents = [-grid_half_length_x, -grid_half_length_y, min(-grid_half_length_x, -grid_half_length_y),
                           grid_half_length_x, grid_half_length_y, grid_height] # xMin,yMin,zMin,xMax,yMax,zMax
        else:
            grid_extents = [-grid_half_length_x, -grid_half_length_y, grid_bottom,
                           grid_half_length_x, grid_half_length_y, grid_height]  # xMin,yMin,zMin,xMax,yMax,zMax
        return cls(grid_extents, cell_size)

    def _set_extents(self,extents):
        extents = np.array(extents).copy().astype(np.float64).round(5).flatten()
        ncells = (extents[3:] - extents[0:3]) // self.cell_size
        extents[3] = extents[0] + ncells[0] * self.cell_size
        extents[4] = extents[1] + ncells[1] * self.cell_size
        extents[5] = extents[2] + ncells[2] * self.cell_size
        self.extents = extents

    def calculate_eigenvalues(self,points) -> None:
        """Calculate the eigenvalues of points within each grid cell. Needs review."""
        pointBins = (points // self.cell_size).astype(np.int32)
        #sortedIndices = pointBins.view([('x', np.float64), ('y', np.float64), ('z', np.float64)]).argsort(axis=0,order=['z','y','x']).flatten()
        #pointBins = pointBins[sortedIndices]
        #points = points[sortedIndices]
        occupiedBins, pointCount = np.unique(pointBins,axis=0,return_counts=True)
        occupiedBins = occupiedBins[pointCount>=3]

        # Create empty array to fill out
        gridBins = (self.centers // self.cell_size).astype(np.int32)

        # Calculate variables used to find correct index of array using bin numbers
        binMin = gridBins.min(axis=0)
        binMax = gridBins.max(axis=0)
        binRange = binMax - binMin + 1

        @jit([void(float64[:, :], int32[:, :], int32[:, :], int32[:], int32[:], float64[:, :])],nopython=True, parallel=True)
        #@guvectorize([(float64[:,:],int32[:,:],int32[:,:], int32[:], int32[:],float64[:,:])],'(a,b),(a,b),(c,d),(e),(e)->(c,d)',nopython=True,target='parallel')
        def calculate_eigenvalues_numba(points, pointBins, occupiedBins, binMin,binRange, geometricFeatures):

            for i in prange(occupiedBins.shape[0]):
                # For each bin (grid cell) in the grid, select all of the points in that bin
                subset = points[(pointBins[:,2] == occupiedBins[i,2]) & (pointBins[:, 1] == occupiedBins[i,1]) & (pointBins[:, 0] == occupiedBins[i,0])]
                gridIndex = (occupiedBins[i,2] - binMin[2]) * binRange[1] * binRange[0] + (occupiedBins[i,1] - binMin[1]) * binRange[0] + (occupiedBins[i,0] - binMin[0])
                eigvals,eigvectors = np.linalg.eig(np.cov(subset.T))
                geometricFeatures[gridIndex,0] = (eigvals[0]**.5-eigvals[1]**.5)/eigvals[0]**.5 # Linearity
                geometricFeatures[gridIndex,1] = (eigvals[1]**.5-eigvals[2]**.5)/eigvals[0]**.5 # Planarity
                geometricFeatures[gridIndex,2] = -(eigvals[0] * np.log(eigvals[0]) + eigvals[1] * np.log(eigvals[1]) + eigvals[2] * np.log(eigvals[2])) # Eigenentropy
                geometricFeatures[gridIndex,3] = 1-abs(eigvectors[2,2]) # Verticality

        calculate_eigenvalues_numba(points, pointBins, occupiedBins, binMin, binRange, self.geometric_features)

    def calculate_pulse_metrics(self,pulses:'Pulses', G = .5) -> None:
        """Trace lidar pulses through grid to estimate PAD and other metrics"""

        # Define function for voxel traversal algorithm, implemented in parallel with numba just-in-time compiler
        @njit([void(float64[:, :], float64, float64[:], float64[:, :, :], float64[:, :, :], float64[:, :, :])],parallel=True)
        def voxel_traversal(pulses, cellSize, gridExtents, pDirected, pTransmitted, pIntercepted):
            """Trace lidar pulses through grid to count number of pulses directed, transmitted, and intercepted
            Here, pulses must be a numpy array with columns """

            for i in prange(pulses.shape[0]):

                # Find which cell the ray ends in
                cellXEnd = np.uint16((pulses[i, 3] - gridExtents[0]) // cellSize)
                cellYEnd = np.uint16((pulses[i, 4] - gridExtents[1]) // cellSize)
                cellZEnd = np.uint16((pulses[i, 5] - gridExtents[2]) // cellSize)

                # Find which cell the ray starts in
                xstart = pulses[i,0]
                ystart = pulses[i,1]
                zstart = pulses[i,2]

                if not ((xstart >= gridExtents[0]) & (ystart >= gridExtents[1]) & (zstart >= gridExtents[2]) &
                        (xstart <= gridExtents[3]) & (ystart <= gridExtents[4]) & (zstart <= gridExtents[5])):

                    # Origin outside bounds. Try to calculate intersection with grid.

                    xdir = pulses[i, 6]
                    ydir = pulses[i, 7]
                    zdir = pulses[i, 8]

                    # Initialize t1 and t2 to handle intersection range
                    if xdir != 0:
                        t_min = (gridExtents[0] - xstart) / xdir
                        t_max = (gridExtents[3] - xstart) / xdir
                    else:
                        t_min = -np.inf
                        t_max = np.inf

                    t1 = min(t_min, t_max)
                    t2 = max(t_min, t_max)

                    # Calculate intersection range for y axis
                    if ydir != 0:
                        t_min = (gridExtents[1] - ystart) / ydir
                        t_max = (gridExtents[4] - ystart) / ydir
                    else:
                        t_min = -np.inf
                        t_max = np.inf

                    t1 = max(t1, min(t_min, t_max))  # Update t1 to the maximum lower bound
                    t2 = min(t2, max(t_min, t_max))  # Update t2 to the minimum upper bound

                    if t1 > t2:
                        continue  # No intersection

                    # Calculate intersection range for z axis
                    if zdir != 0:
                        t_min = (gridExtents[2] - zstart) / zdir
                        t_max = (gridExtents[5] - zstart) / zdir
                    else:
                        t_min = -np.inf
                        t_max = np.inf

                    t1 = max(t1, min(t_min, t_max))  # Update t1 to the maximum lower bound
                    t2 = min(t2, max(t_min, t_max))  # Update t2 to the minimum upper bound

                    if t1 > t2:
                        continue  # No intersection

                    # Update starting coordinate such that it is on the edge of the grid
                    xstart = np.float64(xstart + t1 * xdir)
                    ystart = np.float64(ystart + t1 * ydir)
                    zstart = np.float64(zstart + t1 * zdir)

                if not ((xstart >= gridExtents[0]) & (ystart >= gridExtents[1]) & (zstart >= gridExtents[2]) &
                        (xstart <= gridExtents[3]) & (ystart <= gridExtents[4]) & (zstart <= gridExtents[5])):
                    continue

                cellX = np.uint32((xstart - gridExtents[0]) // cellSize)
                cellY = np.uint32((ystart - gridExtents[1]) // cellSize)
                cellZ = np.uint32((zstart - gridExtents[2]) // cellSize)

                # Calculate tmax as the number of timesteps to reach edge of next voxel.
                # Account for travelling towards upper bounds of voxels or lower bounds of voxels

                if pulses[i, 6] > 0.0:
                    stepX = 1
                    tDeltaX = cellSize / pulses[i, 6]
                    tMaxX = ((gridExtents[0] + (cellX+1) * cellSize) - xstart) / pulses[i, 6]  # X position of voxel boundary - current position of ray / distance along x travelled in each timestep
                elif pulses[i, 6] < 0.0:
                    stepX = -1
                    tDeltaX = cellSize / -pulses[i, 6]
                    tMaxX = ((gridExtents[0] + (cellX) * cellSize) - xstart) / pulses[
                        i, 6]  # -1 from cellX to calculate distance towards left edge of cell instead of right edge of cell
                else:
                    stepX = 0
                    tDeltaX = gridExtents[3]
                    tMaxX = gridExtents[3]

                if pulses[i, 7] > 0.0:
                    stepY = 1
                    tDeltaY = cellSize / pulses[i, 7]
                    tMaxY = ((gridExtents[1] + (cellY+1) * cellSize) - ystart) / pulses[
                        i, 7]  # Y position of voxel boundary - current position of ray / distance along y travelled in each timestep
                elif pulses[i, 7] < 0.0:
                    stepY = -1
                    tDeltaY = cellSize / -pulses[i, 7]
                    tMaxY = ((gridExtents[1] + (cellY) * cellSize) - ystart) / pulses[
                        i, 7]  # -1 from cellY to calculate distance towards down edge of cell instead of up edge of cell
                else:
                    stepY = 0
                    tDeltaY = gridExtents[4]
                    tMaxY = gridExtents[4]

                if pulses[i, 8] > 0.0:
                    stepZ = 1
                    tDeltaZ = cellSize / pulses[i, 8]
                    tMaxZ = ((gridExtents[2] + (cellZ+1) * cellSize) - zstart) / pulses[
                        i, 8]  # Y position of voxel boundary - current position of ray / distance along y travelled in each timestep
                elif pulses[i, 8] < 0.0:
                    stepZ = -1
                    tDeltaZ = cellSize / -pulses[i, 8]
                    tMaxZ = ((gridExtents[2] + (cellZ) * cellSize) - zstart) / pulses[
                        i, 8]  # -1 from cellZ to calculate distance towards down edge of cell instead of up edge of cell
                else:
                    stepZ = 0
                    tDeltaZ = gridExtents[5]
                    tMaxZ = gridExtents[5]

                # Repeat loop until traversing out of the grid along some dimension
                intercepted = False
                while (cellX != pDirected.shape[0]) and (cellY != pDirected.shape[1]) and (
                        cellZ != pDirected.shape[2]) and (cellX != -1) and (cellY != -1) and (cellZ != -1):
                    # Update counters
                    pDirected[cellX, cellY, cellZ] += pulses[i,9]  # Pulse was directed at this voxel, may or may not have passed through

                    if (not intercepted) and (cellZ == cellZEnd) and (cellX == cellXEnd) and (cellY == cellYEnd):
                        pIntercepted[cellX, cellY, cellZ] += pulses[i,9]  # Pulse intercepted within this voxel
                        intercepted = True

                    if not intercepted:
                        pTransmitted[cellX, cellY, cellZ] += pulses[i,9]  # Pulse passed through this voxel

                    if (tMaxX < tMaxY) and (tMaxX < tMaxZ):
                        # X-axis traversal.
                        cellX += stepX
                        tMaxX += tDeltaX
                    elif (tMaxY < tMaxZ):
                        # Y-axis traversal.
                        cellY += stepY
                        tMaxY += tDeltaY
                    else:
                        # Z-axis traversal.
                        cellZ += stepZ
                        tMaxZ += tDeltaZ

        voxel_traversal(pulses.array, self.cell_size, self.extents, self.p_directed, self.p_transmitted, self.p_intercepted)

        meanPathLength = .843*self.cell_size # Correction for unequal path lengths from Grau et al 2017
        self.occlusion = 1 - (self.p_intercepted + self.p_transmitted) / self.p_directed
        self.occlusion[~np.isfinite(self.occlusion)] = 1
        self.pad = -np.log(1 - (self.p_intercepted / (self.p_intercepted + self.p_transmitted))) / (G * meanPathLength)
        self.occlusion[~np.isfinite(self.pad)] = 1
        self.pad[~np.isfinite(self.pad)] = 0

    def add_pulse_metrics(self,grid, G = .5) -> None:
        """Combine pulse metrics from two Grid objects. Grids must overlap exactly (same extents, same cell size)."""
        self.p_directed += grid.p_directed
        self.p_intercepted += grid.p_intercepted
        self.p_transmitted += grid.p_transmitted
        meanPathLength = .843 * self.cell_size  # From Grau et al 2017
        self.pad = -np.log(1 - (self.p_intercepted / (self.p_intercepted + self.p_transmitted))) / (G * meanPathLength)
        self.occlusion = np.minimum(self.occlusion,grid.occlusion)
        self.occlusion[~np.isfinite(self.occlusion)] = 1
        self.occlusion[~np.isfinite(self.pad)] = 1
        self.pad[~np.isfinite(self.pad)] = 0

    def filter_pad_noise_ivf(self, window_radius=15, min_count_present=5) -> None:
        """Search a moving window around each grid cell, set PAD to 0 if cell has few neighbors

        Window size is total width of window with cells as the unit. Must be odd number?
        """
        from scipy import ndimage
        presence = (self.pad>0).astype(np.float32)
        count = ndimage.uniform_filter(presence, window_radius) * window_radius ** 2
        self.pad[count < min_count_present] = 0


    def create_dem_decreasing_window(self, pulses:'Pulses', window_sizes:Sequence[float] = [5, 2.5, 1, .5],
                                     height_thresholds:Sequence[float]=[2.5, 1.25, .5, .25]) -> None:
        """Create a digital elevation model from a point cloud using iterative height filtering algorithm

        Algorithm is based on Caster et al. 2021 https://doi.org/10.1016/j.geoderma.2021.115369

        Args:
            pulses (Pulses): input point cloud. Can include mix of ground and veg, does not need to be pre-classified.
            window_sizes: list of progressively smaller xy window resolutions
            height_thresholds: list of progressively smaller height thresholds

        Returns: None

        """
        import polars as pl
        import scipy
        from voxelmon.utils import interp2D_w_cubic_extrapolation


        try:
            points = pulses.xyz
        except:
            points = pulses[:,0:3]

        # Get lowest point in base grid
        grid = self.bin2D(points,pl.min('z'))
        grid[np.isnan(grid)]=99
        centers = self.centers_xy
        # Create a grid of coordinates corresponding to the array indices
        x, y = np.indices(grid.shape)

        points = pl.DataFrame({'x':points[:,0],'y':points[:,1],'z':points[:,2],'ground':-2})
        points = points.with_columns(pl.col('x').floordiv(self.cell_size).cast(pl.Int32).alias('xBin'),
                                     pl.col('y').floordiv(self.cell_size).cast(pl.Int32).alias('yBin'))

        # Get lowest point in decreasing windows
        for windowSize,heightThresh in zip(window_sizes, height_thresholds):
            # Get low points in moving window
            cellRadius = round(windowSize / 2 / self.cell_size)
            windowShape = [cellRadius*2+1,cellRadius*2+1]
            grid = scipy.ndimage.percentile_filter(grid,.25,windowShape,mode='nearest')

            # Interpolate remaining missing values
            maskMissing = grid == 99
            pointsValid = np.array((x[~maskMissing], y[~maskMissing])).T
            valuesValid = grid[~maskMissing]
            pointsMissing = np.array((x[maskMissing], y[maskMissing])).T
            #grid[maskMissing] = scipy.interpolate.griddata(pointsValid, valuesValid, pointsMissing, method='linear')
            grid[maskMissing] = interp2D_w_cubic_extrapolation(pointsValid, valuesValid, pointsMissing)

            # Relate elevation data to point cloud
            dem_df = pl.DataFrame({'xBin':(centers[:,0] // self.cell_size).astype(np.int32),
                                   'yBin':(centers[:,1] // self.cell_size).astype(np.int32),
                                   'ground':grid.flatten()})
            points = points.drop('ground').join(dem_df,['xBin','yBin'],'left')

            # Remove points that are not near the ground and recalculate lowest points
            points = points.filter(pl.col('z') <= pl.col('ground').add(heightThresh))
            grid = self.bin2D(points, pl.min('z'))
            grid[np.isnan(grid)] = 99

        # Interpolate remaining missing values
        maskMissing = grid == 99
        pointsValid = np.array((x[~maskMissing], y[~maskMissing])).T
        valuesValid = grid[~maskMissing]
        pointsMissing = np.array((x[maskMissing], y[maskMissing])).T
        #grid[maskMissing] = scipy.interpolate.griddata(pointsValid, valuesValid, pointsMissing, method='linear')
        grid[maskMissing] = interp2D_w_cubic_extrapolation(pointsValid, valuesValid, pointsMissing)

        self.dem = grid
        self.hag = self.centers[:,2] - np.tile(grid.flatten(),self.shape[2])

    def calculate_dem_metrics(self) -> dict:
        """Summarize overall terrain slope, aspect, roughness, and concavity"""
        import statsmodels.api as sm
        import pandas as pd
        results = {}

        dem_df = pd.DataFrame({'x': self.centers_xy[:, 0], 'y': self.centers_xy[:, 1], 'z':self.dem.flatten()})

        # Calculate horizontal distance from center
        dem_df['HD'] = np.sqrt(dem_df['x'] ** 2 + dem_df['y'] ** 2)
        # Filter outside of plot radius
        dem_df = dem_df[dem_df['HD'] < 11.3]
        dem_df = dem_df[~np.isnan(dem_df['z'])]
        dem_df['intercept'] = 1

        # Fit a plane using linear regression
        model = sm.OLS(dem_df['z'],dem_df[['intercept','x', 'y']]).fit()

        # Extract coefficients
        intercept = model.params.iloc[0]
        coef_x = model.params.iloc[1]
        coef_y = model.params.iloc[2]

        # Normal vector of the plane
        normal_vector = np.array([coef_x, coef_y, -1])

        # Normalize the normal vector to get a unit vector
        normal_unit_vector = normal_vector / np.linalg.norm(normal_vector)
        if normal_unit_vector[2] < 0:
            normal_unit_vector *= -1

        # Unit vector along the Z-axis (to get terrain slope relative to up)
        z_axis_vector = np.array([0, 0, 1])

        # Unit vector along Y-axis (to get terrain aspect relative to north)
        y_axis_vector = np.array([0, 1, 0])

        # Calculate the dot product between the unit vectors
        dot_product_z = np.dot(normal_unit_vector, z_axis_vector)

        # Assign angles to data
        results['terrain_slope'] = np.degrees(np.arccos(dot_product_z))
        terrain_aspect = np.degrees(np.arctan2(normal_unit_vector[0], normal_unit_vector[1]))
        if terrain_aspect < 0:
            terrain_aspect += 360
        results['terrain_aspect'] = terrain_aspect

        # Calculate terrain shape metrics
        dem_df['resid'] = dem_df['z'] - model.predict(dem_df[['intercept', 'x', 'y']])
        hd_half = 11.3 / 2
        sum_inner = dem_df[dem_df['HD'] < hd_half]['resid'].sum()
        sum_outer = dem_df[dem_df['HD'] >= hd_half]['resid'].sum()
        results['terrain_concavity'] = sum_inner - sum_outer
        results['terrain_roughness'] = np.sqrt(np.mean(dem_df['resid'] ** 2))
        return results

    def calculate_canopy_cover(self, clip_radius:Union[float,None] = 11.3,
                               cutoff_height:float = 2,
                               max_occlusion:float = .8) -> float:
        """Calculate canopy cover using raster approach

        Args:
            clip_radius (float) or None: clip to circular plot with specified radius. If None, does not clip.
            cutoff_height: minimum height of vegetation considered in canopy cover
            max_occlusion: nullifies raster cells where mean occlusion is greater than this value.

        """
        import polars as pl
        df = self.to_polars().select(['x','y','hag','pad','occlusion'])
        df = df.filter(pl.col('hag').ge(cutoff_height))
        if clip_radius is not None:
            df = df.filter(((pl.col('x')**2 + pl.col('y')**2)**.5).le(clip_radius))
        df = df.group_by((pl.col('x') / self.cell_size).cast(int), (pl.col('y') / self.cell_size).cast(int)) \
               .agg([(pl.col("pad") >= 0.05).sum().alias('filled'),
                     pl.col("occlusion").mean().alias("occlusion")])
        total_cells = df.filter(pl.col('filled').gt(0) | pl.col('occlusion').le(max_occlusion)).shape[0]
        filled_cells = df.filter(pl.col('filled').gt(0)).shape[0]
        return filled_cells / total_cells

    def classify_foliage_with_PAD(self, max_occlusion:float=.8, min_pad_foliage=.05, max_pad_foliage=6) -> None:
        """Classify voxels based on PAD thresholds and occlusion value.

        Sets self.classification and self.classification_key.
        Key: {-2: 'empty', -1: 'occluded', 3: 'foliage', 5: 'nonfoliage'}

        Args:
            max_occlusion: if occlusion is greater than this value, classify as occluded
            min_pad_foliage: if PAD is less than this value, classify as empty, otherwise classify foliage
            max_pad_foliage: if PAD is greater than this value, classify as non-foliage
        """
        self.classification[:,:,:] = -2
        self.classification[self.occlusion > max_occlusion] = -1
        self.classification[self.pad<0] = -1
        self.classification[self.pad >= min_pad_foliage] = 3
        self.classification[self.pad > max_pad_foliage] = 5
        self.classification_key = {-2: 'empty', -1: 'occluded', 3: 'foliage', 5: 'nonfoliage'}

    def gaussian_filter_PAD(self,sigma=.1):
        """Apply gaussian smoothing filter to PAD grid"""

        from scipy import ndimage,interpolate
        # Interpolate missing values
        #occlusionFlat = self.occlusion.flatten()
        #pointsValid = self.centers[occlusionFlat<=maxOcclusion]
        #valuesValid = occlusionFlat[occlusionFlat<=maxOcclusion]
        #pointsMissing = self.centers[occlusionFlat>maxOcclusion]
        #vals = interpolate.griddata(pointsValid, valuesValid, pointsMissing, method='linear')
        #self.pad[self.occlusion>maxOcclusion] = vals.reshape(self.shape)

        self.pad = ndimage.gaussian_filter(self.pad,sigma)


    def bin2D(self,pulses,function)->'pl.DataFrame':
        """Aggregate values from pulses to 2D grid

        Function should be from polars and should specify a column name x, y, or z, e.g. pl.min('z')"""
        import polars as pl
        try:
            points_df = pl.DataFrame({'x':pulses.xyz[:,0],'y':pulses.xyz[:,1],'z':pulses.xyz[:,2]})
        except:
            points_df = pl.DataFrame({'x':pulses[:,0],'y':pulses[:,1],'z':pulses[:,2]})

        points_df = points_df.with_columns(pl.col('x').floordiv(self.cell_size).cast(pl.Int32).alias('xBin'),
                                           pl.col('y').floordiv(self.cell_size).cast(pl.Int32).alias('yBin'))

        bins_df = pl.DataFrame({'xBin': (self.centers_xy[:, 0] // self.cell_size).astype(np.int32),
                                'yBin': (self.centers_xy[:, 1] // self.cell_size).astype(np.int32)})

        binVals = points_df.drop_nulls().group_by(['xBin','yBin']).agg(function)

        return bins_df.join(binVals,['xBin','yBin'],'left').to_numpy()[:,2].reshape([self.shape[0],self.shape[1]],order='c')

    def to_polars(self)->'pl.DataFrame':
        """Convert grid to polars dataframe"""
        import polars
        df = polars.DataFrame({'x': self.centers[:, 0], 'y': self.centers[:, 1], 'z': self.centers[:, 2], 'hag':self.hag,
                                'pDirected': self.p_directed.flatten('F'), 'pTransmitted': self.p_transmitted.flatten('F'),
                                'pIntercepted': self.p_intercepted.flatten('F'), 'occlusion': self.occlusion.flatten('F'),
                                'pad': self.pad.flatten('F'), 'linearity': self.geometric_features[:, 0],
                                'planarity': self.geometric_features[:, 1], 'eigenentropy': self.geometric_features[:, 2],
                                'verticality': self.geometric_features[:, 3], 'classification':self.classification.flatten('F')})
        return df


    def summarize_by_height(self, clip_radius:Union[float,None] = None,
                            foliage_classes:Sequence[int] = [3],
                            empty_class:int=-2) -> 'pl.DataFrame':
        """Summarize grid within bins of height-above-ground

        Estimates mean PAD including foliage classes and empty_class while ignoring other classes

        Args:
            clip_radius (float or None): if not None, will clip to circular plot based on origin and clip_radius
            foliage_classes (Sequence[int]): class ID values that should be considered foliage
            empty_class (int): class ID value that should be considered empty

        """
        import polars as pl

        df = self.to_polars()

        if clip_radius!=None:
            df = df.filter(((pl.col('x')**2 + pl.col('y')**2)**.5) < clip_radius)

        # Filter bad height data
        df = df.filter((pl.col('hag')>=0) & (pl.col('hag').is_not_nan()))

        # Assign height bins
        df = df.with_columns(pl.col('hag').floordiv(self.cell_size).cast(pl.Int32).alias('heightBin'))

        # Count voxels by class and height bin
        summary = df.pivot(on="classification", index="heightBin", values='classification', aggregate_function=pl.len())
        summary = summary.fill_null(0)

        # Assign meaningful column names
        summary.columns =['heightBin'] + [self.classification_key[int(col)] for col in summary.columns[1:]]

        if 'occluded' not in summary.columns:
            summary = summary.with_columns(pl.lit(0.0).alias('occluded'))

        # Convert count to proportion of non-occluded volume
        volNO = summary.drop(['heightBin','occluded']).sum_horizontal()
        volTotal = summary.drop(['heightBin']).sum_horizontal()
        pctOccluded = summary['occluded'] / volTotal
        summary = summary.with_columns(summary.drop(['heightBin']) / volNO)
        summary = summary.with_columns(pctOccluded)


        # Get mean PAD within foliage and empty space, ignoring occluded areas and other classes
        pad = df.filter(pl.col('classification').is_in(foliage_classes + [empty_class]))
        pad = pad.group_by('heightBin').agg(pl.mean('pad')).sort('heightBin')

        summary = summary.join(pad,on='heightBin',how='full',coalesce=True).sort('heightBin')

        # Clean up
        summary = pl.DataFrame((summary['heightBin'].cast(pl.Float64) * (self.cell_size)).alias('height')).hstack(summary)
        summary = summary.fill_null(0).fill_nan(0)

        return summary

    def export_grid_as_csv(self, filepath):
        """Write grid to csv file (compatible with CloudCompare and other software)"""
        df = self.to_polars()
        df.write_csv(filepath)

    def export_dem_as_csv(self,filepath):
        """Write terrain model to points-like csv file (compatible with CloudCompare)"""
        import polars
        df = polars.DataFrame({'x': self.centers_xy[:, 0], 'y': self.centers_xy[:, 1], 'z':self.dem.flatten()})
        df.write_csv(filepath)

class Pulses:
    def __init__(self,pulses_df:'pl.DataFrame'):
        """Initialize Pulses from dataframe

        Required columns:
            X (double): X coordinate of return
            Y (double): Y coordinate of return
            Z (double): Z coordinate of return
            BeamOriginX (double): X coordinate of beam origin
            BeamOriginY (double): Y coordinate of beam origin
            BeamOriginZ (double): Z coordinate of beam origin

        Columns added if not present:
            BeamDirectionX (double): Beam direction unit vector X coordinate
            BeamDirectionY (double): Beam direction unit vector X coordinate
            BeamDirectionZ (double): Beam direction unit vector X coordinate
            Weight (double): Weight of this return used in path tracing calculations (defaults to 1/NumberOfReturns or 1)
            Classification (Uint8): Point classification label

        """
        required_cols = ['X', 'Y', 'Z', 'BeamOriginX', 'BeamOriginY', 'BeamOriginZ']
        for col in required_cols:
            if col not in pulses_df.columns:
                raise KeyError(f'Required column {col} not found in pulses_df')

        self.df = pulses_df

        if 'BeamDirectionX' not in pulses_df.columns or 'BeamDirectionY' not in pulses_df.columns or 'BeamDirectionZ' not in pulses_df.columns:
            self.calculate_unit_vectors()

        if 'Weight' not in pulses_df.columns:
            if 'NumberOfReturns' in self.df.columns:
                self.df = self.df.with_columns(pl.lit(1).truediv(pl.col('NumberOfReturns')).alias('Weight'))
            else:
                self.df = self.df.with_columns(pl.lit(1).alias('Weight'))

        if 'Classification' not in pulses_df.columns:
            self.df = self.df.with_columns(pl.lit(0).alias('Classification'))

    @classmethod
    def from_point_cloud_array(cls, xyz_array:Union['np.array','pl.DataFrame'],
                               origin:Union[Sequence,'np.array','pl.DataFrame'] = None,):
        """Initialize from numpy array or pl.DataFrame where the first three columns are x, y, z coordinates

        Origin must be provided as sequence of three values (x, y, z) for single origin or as matrix where first three
        columns represent x, y, z coordinates of origin and number of rows matches xyz_array
        """
        origin = np.array(origin)
        if len(origin.shape) == 1:
            df = pl.DataFrame({'X': xyz_array[:,0], 'Y': xyz_array[:,1], 'Z': xyz_array[:,2],
                               'BeamOriginX': origin[0], 'BeamOriginY': origin[1], 'BeamOriginZ': origin[2]})
        else:
            df = pl.DataFrame({'X': xyz_array[:,0], 'Y': xyz_array[:,1], 'Z': xyz_array[:,2],
                               'BeamOriginX': origin[:,0], 'BeamOriginY': origin[:,1], 'BeamOriginZ': origin[:,2]})
        try:
            df = df.with_columns(pl.lit(xyz_array['Classification']).alias('Classification'))
        except:
            pass

        try:
            df = df.with_columns(pl.lit(xyz_array['NumberOfReturns']).alias('NumberOfReturns'))
        except:
            pass

        return cls(df)

    @classmethod
    def from_file(cls, filepath, origin=None):
        import utils
        df = utils.open_file_pdal(filepath)
        if origin is None:
            return cls(df)
        else:
            return cls.from_point_cloud_array(df,origin)

    @property
    def array(self):
        return self.df.select(['BeamOriginX', 'BeamOriginY', 'BeamOriginZ', 'X', 'Y', 'Z',
                               'BeamDirectionX', 'BeamDirectionY', 'BeamDirectionZ', 'Weight']).to_numpy()

    @property
    def xyz(self) -> 'np.array':
        return self.df.select(['X','Y','Z']).to_numpy()

    def calculate_unit_vectors(self):
        arr = self.df.select(['BeamOriginX', 'BeamOriginY', 'BeamOriginZ', 'X', 'Y', 'Z'])
        diff = arr[:, 3:6] - arr[:, 0:3]
        length = (diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2) ** .5  # calculate total length
        self.df = self.df.with_columns(pl.lit(diff[:, 0] / length).alias('BeamDirectionX'))   # calculate x component of unit vector
        self.df = self.df.with_columns(pl.lit(diff[:, 1] / length).alias('BeamDirectionY'))  # calculate y component of unit vector
        self.df = self.df.with_columns(pl.lit(diff[:, 2] / length).alias('BeamDirectionZ'))   # calculate z component of unit vector

    def crop(self,extents)->'Pulses':
        mask = ((self.df['X']>=extents[0]) & (self.df['Y']>=extents[1]) &
               (self.df['Z']>=extents[2]) & (self.df['X']<=extents[3]) &
               (self.df['Y']<=extents[4]) & (self.df['Z']<=extents[5]))

        return Pulses(self.df.filter(mask))

    def thin_distance_weighted_random(self,propRemaining)->'Pulses':
        arr = self.array
        diff = arr[:, 3:6] - arr[:, 0:3]
        distsq = (diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2)
        weights = distsq / distsq.sum()
        randIndex = np.random.choice(distsq.size,int(distsq.size * propRemaining),replace=False,p=weights)
        return Pulses(self.df.filter(randIndex))

    def to_csv(self, filepath, coordinates_only=True):
        """Write pulses to csv file. Optionally, export return coordinates only."""
        if coordinates_only:
            self.df.select(['X','Y','Z']).write_csv(filepath)
        else:
            self.df.write_csv(filepath)

class ALS:
    def __init__(self,filepath, bounds:str = None):
        """Initialize ALS reader

        Args:
            filepath (str): Path to ALS file readable by pdal. Type is inferred by extension.
            bounds (str): Clip extents of the resource in 2 or 3 dimensions, formatted as pdal-compatible string,
                e.g.: ([xmin, xmax], [ymin, ymax], [zmin, zmax]). If omitted, the entire dataset will be selected.
                The bounds can be followed by a slash (‘/’) and a spatial reference specification to apply to the bounds.
            """
        from voxelmon.utils import open_file_pdal
        self.path = filepath
        self.bounds = bounds
        self.points, self.crs = open_file_pdal(self.path, self.bounds)

    def estimate_flightpath(self, min_separation:float=2,
                            time_bin_size:float=.5,
                            fit_line:bool=True,
                            min_z_q:float = .75,
                            flightline_t_break:float=5)->'pl.DataFrame':
        """Estimate flightpath by triangulating position from rays drawn between first and last return

        This implementation is optimized for tiled data. It assumes that each flight path can be represented by a
        straight line at a constant altitude above MSL (not ground).

        Args:
            min_separation (float): Minimum separation distance between first and last return considered when drawing rays
            time_bin_size (float): Time bin size used when estimating current plane position
            fit_line (bool): If True, fits a straight line to each flight path.
                If False, returns all estimates of current location.
            min_z_q (float): Clip location estimates to only include z values above this quantile of z
            flightline_t_break (float): Separates returns to a new flightline if there is a gap in GPS time between
                returns longer than this threshold."""
        import polars
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        from sklearn.linear_model import LinearRegression
        from voxelmon.utils import interpolate_flightpath

        self.points = self.points.sort(['GpsTime','ReturnNumber'])

        def split_flightlines(points):
            gps_time_arr = points['GpsTime'].to_numpy()
            point_views = []
            prev_i = 0
            for i in range(1,len(gps_time_arr)):
                if (gps_time_arr[i] - gps_time_arr[i-1]) > flightline_t_break:
                    point_views.append(points[prev_i:i,:])
                    prev_i = i
            point_views.append(points[prev_i:,:])
            return point_views

        points_views = split_flightlines(self.points)

        def get_convergence_points(points):
            points_all = points[['X','Y','Z']].to_numpy()
            return_number_all = points['ReturnNumber'].to_numpy()
            gps_time = points['GpsTime'].to_numpy()
            start_time = gps_time.min()
            end_time = gps_time.max()
            flight_points = []
            for time in np.arange(start_time,end_time+time_bin_size, time_bin_size):
                time_mask = (gps_time >= time) & (gps_time < time + time_bin_size)
                points_arr = points_all[time_mask]
                return_number = return_number_all[time_mask]
                first = []
                last = []
                count = 0
                first_temp = np.array([np.nan,np.nan,np.nan])
                for i in range(len(return_number)):
                    if return_number[i] == 1:
                        if count > 1:
                            first.append(first_temp)
                            last.append(points_arr[i-1,:])
                        first_temp = points_arr[i,:]
                        count = 1
                    else:
                        count += 1

                if len(first) > 1:
                    first = np.array(first)
                    last = np.array(last)

                    dif = first - last
                    length = (dif ** 2).sum(1) ** .5
                    dif = dif[length>=min_separation,:]
                    origin = last[length>=min_separation,:]
                    length = length[length>=min_separation]
                    dir = dif / np.repeat(length,3).reshape([len(length),3])
                    extreme_z = np.quantile(dir[:,2], [.1,0.9])
                    extreme_mask = (dir[:,2] <= extreme_z[0]) | (dir[:,2] >= extreme_z[1])
                    origin_extreme = origin[extreme_mask,:]
                    dir_extreme = dir[extreme_mask,:]

                    A = np.zeros((3, 3))
                    b = np.zeros(3)

                    for o, d in zip(origin_extreme, dir_extreme):
                        d = d.reshape(3, 1)
                        P = np.eye(3) - d @ d.T  # Projection matrix
                        A += P
                        b += P @ o

                    p, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                    p = np.hstack([p,time])
                    flight_points.append(p)
            points = np.stack(flight_points)
            points = points[points[:,2]>=np.quantile(points[:,2],min_z_q),:]
            if fit_line:
                lmx = LinearRegression().fit(points[:,3].reshape(-1,1),points[:, 0].reshape(-1, 1))
                lmy = LinearRegression().fit(points[:, 3].reshape(-1, 1), points[:, 1].reshape(-1, 1))
                #lmz = LinearRegression().fit(points[:, 3].reshape(-1, 1), points[:, 2].reshape(-1, 1))
                gps_time_out = np.linspace(gps_time.min(),gps_time.max(),100).reshape(-1,1)
                x = lmx.predict(gps_time_out)
                y = lmy.predict(gps_time_out)
                #z = lmz.predict(gps_time_out)
                z = np.ones_like(x) * points[:, 2].mean()
                points = np.concatenate([x,y,z,gps_time_out],axis=1)


            return points

        with ThreadPoolExecutor() as executor:
            flightlines = list(executor.map(get_convergence_points, points_views))


        for i in range(len(flightlines)):
            flight_points = flightlines[i]
            flightlines[i] = polars.DataFrame({'X':flight_points[:,0],'Y':flight_points[:,1], 'Z':flight_points[:,2],
                                              'GpsTime':flight_points[:,3],'FlightLine':i})

        flightpath = polars.concat(flightlines)
        self.origin = interpolate_flightpath(self.points,flightpath)
        return flightpath

    def quick_lad(self,bin_size_xy = 10, bin_size_z = 2, min_height = 1, extinction_coefficient=.5, return_type='polars'):
        """Estimate leaf area density with assumption that pulses were directed straight down

        return_type must be one of 'numpy', 'polars', or 'xarray'"""
        import pdal
        import numpy as np
        from voxelmon.utils import bin3D
        import polars as pl
        count = 10
        result = 0
        while result == 0 and count <100:
            try:
                if self.bounds is not None:
                    pipeline = pdal.Pipeline([pdal.Reader(self.path,bounds=self.bounds),pdal.Filter.hag_delaunay(count=count)])
                else:
                    pipeline = pdal.Pipeline([pdal.Reader(self.path), pdal.Filter.hag_delaunay(count=count)])
                result = pipeline.execute()
            except:
                count*=2

        arr = pipeline.arrays[0]
        arr = np.column_stack((arr['X'],arr['Y'],arr['HeightAboveGround']))
        arr[arr[:,2]<0,2] = 0 # Assign negative heights to 0

        arr[:,2] -= min_height

        counts = bin3D(arr,pl.len(),[bin_size_xy,bin_size_xy,bin_size_z],asArray=False)
        counts = counts.fill_nan(0).fill_null(0)
        grid_shape = [counts['xBin'].n_unique(),counts['yBin'].n_unique(),counts['zBin'].n_unique()]
        counts_grid = counts[:, -1].to_numpy().reshape(grid_shape, order='f')

        counts_cs = counts_grid.cumsum(axis=2)
        pulses_in = counts_cs[:,:,1:]
        pulses_out = counts_cs[:,:,:-1]
        lad = -np.log(pulses_out/pulses_in)/(extinction_coefficient*bin_size_z)
        lad[pulses_in==0] = -1
        lad[pulses_out==0] = -2

        x_centers = counts['xBin'].unique() * bin_size_xy + bin_size_xy / 2
        y_centers = counts['yBin'].unique() * bin_size_xy + bin_size_xy / 2
        z_centers = counts['zBin'].unique()[1:] * bin_size_z + bin_size_z / 2 + min_height

        if return_type == 'numpy':
            return lad
        elif return_type == 'polars':
            z, y, x = np.meshgrid(z_centers,y_centers,x_centers, indexing='ij')
            return pl.DataFrame({'X':x.flatten(),'Y': y.flatten(), 'Z':z.flatten(),'LAD':lad.flatten('F')})
        elif return_type == 'xarray':
            import xarray as xr
            return xr.DataArray(lad,coords={'X':x_centers,'Y':y_centers,'Z':z_centers},dims=['X','Y','Z'],name='LAD')

    def execute_default_processing(self,export_folder:str,
                                   plot_name:str,
                                   cell_size:float=1,
                                   extents:Union[Sequence[float],None] = None,
                                   max_occlusion=.8,
                                   sigma1=0,
                                   min_pad_foliage=.01,
                                   max_pad_foliage=6):
        from voxelmon.utils import _default_postprocessing, _default_folder_setup

        _default_folder_setup(export_folder)

        # TODO: Write utility function to interpolate origin from flightpath

        pulses = Pulses.from_point_cloud_array(self.points, origin=self.origin)

        if extents is None:
            min_extents = self.points.select(['X','Y','Z']).min().to_numpy()
            max_extents = self.points.select(['X', 'Y', 'Z']).max().to_numpy()
            extents = np.concatenate([min_extents,max_extents]).flatten()

        pulses_thin = pulses.crop(extents)

        grid = Grid(extents=extents, cell_size=cell_size)

        grid.create_dem_decreasing_window(pulses_thin,window_sizes=[5,2.5,1],height_thresholds=[2.5,1.25,.5])

        grid.calculate_pulse_metrics(pulses)

        profile, summary = _default_postprocessing(grid=grid, plot_name=plot_name, export_folder=export_folder,
                                                   plot_radius=None, max_occlusion=max_occlusion, sigma1=sigma1,
                                                   min_pad_foliage=min_pad_foliage, max_pad_foliage=max_pad_foliage)

        return profile, summary


class PtxBlk360G1:
    def __init__(self,filepath,applyTranslation=True,applyRotation=True,dropNull=False,offset=[0,0,0]):
        self.path = filepath
        self.load_points(dropNull=dropNull)
        self.get_transform()
        self.apply_transform(self.transform,applyTranslation=applyTranslation,applyRotation=applyRotation)
        self.apply_offset(offset)
        self.get_polar_coordinates()
        if dropNull==False:
            self.create_pseudo_returns()

    def load_points(self,dropNull=False):
        import polars
        # Get num cols
        firstRow = np.loadtxt(self.path, np.float64, skiprows=10, max_rows=1)
        if firstRow.size==4:
            schema = [polars.Float64] * 4
        elif firstRow.size==7:
            schema = [polars.Float64] * 4 + [polars.Int32]*3
        else:
            raise('Unexpected number of columns in PTX file')

        points = polars.read_csv(self.path, skip_rows=10, has_header=False, separator=' ',schema_overrides=schema).to_numpy().astype(np.float64)

        nRowsCols = np.loadtxt(self.path, skiprows=0, max_rows=2)
        self.ncols = int(nRowsCols[0])
        self.nrows = int(nRowsCols[1])

        rows,cols = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing='xy')

        self.rowsCols = np.vstack([rows.flatten(), cols.flatten()]).T

        nullMask = ((points[:, 0] == 0) & (points[:, 1] == 0) & (points[:, 2] == 0))
        if dropNull:
            points = points[~nullMask]
            self.rowsCols = self.rowsCols[nullMask]
            self.nullMask = ~np.isnan(points[:,0])
        else:
            self.nullMask = nullMask

        self.npoints = points.shape[0]
        self.xyz = points[:,:3]
        self.intensity = points[:,3]
        if points.shape[1] > 4:
            self.rgb = points[:,4:7]
        else:
            self.rgb = self.intensity.repeat(3).reshape([self.npoints, 3])

    def filter(self, boolArray):
        self.xyz = self.xyz[boolArray]
        self.ptrh = self.ptrh[boolArray]
        self.rgb = self.rgb[boolArray]
        self.intensity = self.intensity[boolArray]
        self.rowsCols = self.rowsCols[boolArray]
        self.npoints = self.intensity.size


    def get_transform(self):
        import os
        aux_transform = self.path[:-4] + '.mat.txt'
        if os.path.exists(aux_transform):
            self.transform = np.loadtxt(aux_transform)
        else:
            self.transform = np.loadtxt(self.path,np.float64,skiprows=6,max_rows=4).T
        self.originOriginal = self.transform[:3,3]
        self.origin = np.array([0.,0.,0.])

    def apply_transform(self,transform,applyTranslation=True,applyRotation=True):
        toApply = np.eye(4,4)
        if applyTranslation:
            toApply[:,3] = transform[:,3]
        if applyRotation:
            toApply[:,:3] = transform[:,:3]
        xyzMat = np.hstack([self.xyz,np.ones([self.npoints,1])])
        res = np.matmul(toApply,xyzMat.T).T
        self.xyz = res[:,:3]
        self.xyz = self.xyz.astype(np.float64)
        self.origin += toApply[:3,3]

    def apply_offset(self,xyz):
        self.xyz[:,0] += xyz[0]
        self.xyz[:,1] += xyz[1]
        self.xyz[:,2] += xyz[2]
        self.origin[0] += xyz[0]
        self.origin[1] += xyz[1]
        self.origin[2] += xyz[2]


    def get_polar_coordinates(self):
        xyzTemp = self.xyz.copy()
        xyzTemp[:, 0] -= self.origin[0]
        xyzTemp[:, 1] -= self.origin[1]
        xyzTemp[:, 2] -= self.origin[2]

        self.ptrh = np.zeros([self.npoints,4],np.float64)

        # Calculate polar distances
        self.ptrh[:,2] = np.sqrt(xyzTemp[:,0] ** 2 + xyzTemp[:,1] ** 2 + xyzTemp[:,2] ** 2)
        self.ptrh[:,3] = np.sqrt(xyzTemp[:,0] ** 2 + xyzTemp[:,1] ** 2)

        # Calculate angles

        h1 = xyzTemp[:,2] >= 0
        h2 = xyzTemp[:,2] < 0

        self.ptrh[:, 0][h1] = np.arctan(self.ptrh[:,3][h1] / xyzTemp[:,2][h1])
        self.ptrh[:, 0][h2] = (np.pi / 2) + np.arctan(-xyzTemp[:,2][h2] / self.ptrh[:,3][h2])

        q1 = (xyzTemp[:,0] >= 0) & (xyzTemp[:,1] >= 0)
        q2 = (xyzTemp[:,0] < 0) & (xyzTemp[:,1] >= 0)
        q3 = (xyzTemp[:,0] < 0) & (xyzTemp[:,1] < 0)
        q4 = (xyzTemp[:,0] >= 0) & (xyzTemp[:,1] < 0)

        self.ptrh[:,1][q1] = np.arctan(xyzTemp[:,1][q1] / xyzTemp[:,0][q1])
        self.ptrh[:,1][q2] = (np.pi / 2) + np.arctan(-xyzTemp[:,0][q2] / xyzTemp[:,1][q2])
        self.ptrh[:,1][q3] = np.pi + np.arctan(xyzTemp[:,1][q3] / xyzTemp[:,0][q3])
        self.ptrh[:,1][q4] = (3 * np.pi / 2) + np.arctan(xyzTemp[:,0][q4] / -xyzTemp[:,1][q4])

        # Fill nan=-99 for null returns
        self.ptrh[self.nullMask,:] = -1.

    def to_pandas(self):
        import pandas as pd
        df = pd.DataFrame({'x':self.xyz[:,0],'y':self.xyz[:,1],'z':self.xyz[:,2],'p':self.ptrh[:,0],'t':self.ptrh[:,1],
                           'r':self.ptrh[:,2],'h':self.ptrh[:,3],'row':self.rowsCols[:,0],'col':self.rowsCols[:,1]})
        return df

    def to_polars(self):
        import polars
        df = polars.DataFrame({'x':self.xyz[:,0],'y':self.xyz[:,1],'z':self.xyz[:,2],'p':self.ptrh[:,0],'t':self.ptrh[:,1],
                                'r':self.ptrh[:,2],'h':self.ptrh[:,3],'row':self.rowsCols[:,0],'col':self.rowsCols[:,1]})
        return df

    def to_csv(self,filepath):
        df = self.to_polars()
        df.write_csv(filepath)

    def to_voxelmon_pulses(self):
        return voxelmon.Pulses.from_point_cloud_array(self.xyz,self.origin)

    def create_pseudo_returns(self,pseudoRDValue=99):

        @njit([void(float64[:,:],float64[:,:],int32,int32)],parallel=False)
        def create_pseudo_returns_nb(xyz,ptrh,nrows,ncols):
            for colStart in prange(0,nrows*ncols,nrows):
                colEnd = colStart + nrows
                for i in range(colStart,colEnd):

                    if (ptrh[i,0]<0.1):
                        # If value is missing
                        # Find the next non-missing value
                        j = i + 1
                        while j < colEnd and (ptrh[j,0]<0.):
                            j += 1

                        if j < colEnd:
                            # Linear interpolation
                            start = i - 1
                            end = j
                            if start >= colStart:
                                slopeP = (ptrh[end,0] - ptrh[start,0]) / (end-start)
                                slopeT = (ptrh[end,1] - ptrh[start,1]) / (end-start)
                                for k in range(start + 1, end):
                                    ptrh[k,0] = ptrh[start,0] + slopeP * (k - start)
                                    ptrh[k, 1] = ptrh[start,1] + slopeT * (k - start)
                            else:
                                # If start is -1, it means the missing values are at the start
                                # Use 180 degrees (straight down) for start value of P
                                slopeP = (ptrh[end, 0] - np.pi) / (end - start)
                                for k in range(i, end):
                                    # Fill T with first value
                                    ptrh[k,1] = ptrh[end,1]
                                    # Interpolate P
                                    #ptrh[k, 0] = ptrh[start, 0] + slopeP * (k - start)
                                    # Fill P with straight down
                                    ptrh[k,0] = np.pi
                        else:
                            # No non-missing values found after i
                            if i < colEnd-1:
                                if i == colStart:
                                    # Entire col is empty
                                    for k in range(i, j):
                                        # Straight down
                                        ptrh[k, 0] = np.pi
                                        ptrh[k, 1] = 0
                                else:
                                    start = i - 1
                                    end = j
                                    # Use 0 degrees (straight up) for end value of P
                                    slopeP = (0. - ptrh[i-1,0]) / (j-start)
                                    # Extrapolate T from last pair of points
                                    slopeT = ptrh[i - 1, 1] - ptrh[i - 2, 1]
                                    for k in range(start + 1, end):
                                        ptrh[k, 0] = ptrh[start, 0] + slopeP * (k - start)
                                        ptrh[k, 1] = ptrh[start, 1] + slopeT * (k - start)
                            else:
                                # Final row is missing
                                ptrh[i, 0] = 0
                                ptrh[i, 1] = ptrh[i-1, 1] + slopeT






        create_pseudo_returns_nb(self.xyz,self.ptrh,self.nrows,self.ncols)

        returnMask = ~self.nullMask
        # fill radial distance for non-returns
        self.ptrh[~returnMask, 2] = pseudoRDValue

        # calculate x, y, z for non-returns
        self.xyz[~returnMask, 0] = self.ptrh[~returnMask, 2] * np.sin(self.ptrh[~returnMask, 0]) * np.cos(self.ptrh[~returnMask, 1]) + self.origin[0]
        self.xyz[~returnMask, 1] = self.ptrh[~returnMask, 2] * np.sin(self.ptrh[~returnMask, 0]) * np.sin(self.ptrh[~returnMask, 1]) + self.origin[1]
        self.xyz[~returnMask, 2] = self.ptrh[~returnMask, 2] * np.cos(self.ptrh[~returnMask, 0]) + self.origin[2]
        self.ptrh[~returnMask, 3] = ((self.xyz[~returnMask, 0] - self.origin[0]) ** 2 + (self.xyz[~returnMask, 0] - self.origin[1]) ** 2) ** .5

        # Drop empty starting rows where pseudoreturn is pointed straight down
        keep = self.ptrh[:, 0] < 3.1
        self.filter(keep)

    def execute_default_processing(self, export_folder, plot_name, cell_size=.1, plot_radius=11.3, plot_radius_buffer=.7, max_height=99, max_occlusion=.8, sigma1=0, min_pad_foliage=.01, max_pad_foliage=6):
        import os
        from voxelmon.utils import _default_postprocessing,_default_folder_setup

        _default_folder_setup(export_folder)

        pulses = Pulses.from_point_cloud_array(self.xyz, self.origin)

        maxExtents = [-plot_radius - plot_radius_buffer, -plot_radius - plot_radius_buffer, -plot_radius, plot_radius + plot_radius_buffer, plot_radius + plot_radius_buffer, max_height]

        pulses_thin = pulses.crop(maxExtents)
        minHeight = pulses_thin.xyz[:, 2].min()
        max_height = pulses_thin.xyz[:, 2].max() + 1
        gridExtents = maxExtents.copy()
        gridExtents[2] = minHeight
        gridExtents[5] = max_height

        grid = Grid(extents=gridExtents, cell_size=cell_size)

        grid.create_dem_decreasing_window(pulses_thin)

        grid.calculate_pulse_metrics(pulses)

        profile, summary = _default_postprocessing(grid=grid, plot_name=plot_name, export_folder=export_folder, plot_radius=plot_radius, max_occlusion=max_occlusion, sigma1=sigma1, min_pad_foliage=min_pad_foliage, max_pad_foliage=max_pad_foliage)

        return profile, summary

class PtxBlk360G1_Group:
    def __init__(self,filepathList):
        ptxGroup = [PtxBlk360G1(filepathList[0], applyTranslation=False, applyRotation=True, dropNull=False)]
        offset = -ptxGroup[0].originOriginal
        for ptxFile in filepathList[1:]:
            ptx = PtxBlk360G1(ptxFile, applyTranslation=True, applyRotation=True, dropNull=False, offset=offset)
            ptxGroup.append(ptx)

        self.ptxGroup = ptxGroup

    def execute_default_processing(self, export_folder, plot_name, cell_size=.1, plot_radius=11.3, plot_radius_buffer=.7, max_height=99, max_occlusion=.8, sigma1=.1, min_pad_foliage=.01, max_pad_foliage=6):
        import os
        from voxelmon.utils import _default_folder_setup, _default_postprocessing

        _default_folder_setup(export_folder)

        maxExtents = [-plot_radius - plot_radius_buffer, -plot_radius - plot_radius_buffer, -plot_radius, plot_radius + plot_radius_buffer, plot_radius + plot_radius_buffer, max_height]

        pulsesList = []
        pulsesThinAll = []
        for ptx in self.ptxGroup:
            pulses = Pulses.from_point_cloud_array(ptx.xyz,ptx.origin)
            pulses_thin = pulses.crop(maxExtents).xyz
            pulsesThinAll.append(pulses_thin)
            pulsesList.append(pulses)
        pulsesThinAll = np.concatenate(pulsesThinAll)

        minHeight = pulsesThinAll[:,2].min()
        max_height = pulsesThinAll[:, 2].max() + 1
        extents = maxExtents.copy()
        extents[2] = minHeight
        extents[5] = max_height

        grid = Grid(extents=extents, cell_size=cell_size)

        grid.create_dem_decreasing_window(pulsesThinAll)

        grid.calculate_pulse_metrics(pulsesList[0])

        if len(pulsesList)>1:
            for pulses in pulsesList[1:]:
                grid_temp = Grid(extents=extents, cell_size=cell_size)
                grid_temp.calculate_pulse_metrics(pulses)
                grid.add_pulse_metrics(grid_temp)

        #pulses_thin.to_csv(os.path.join(export_folder, 'Points/', plot_name) + '.csv')

        profile, summary = _default_postprocessing(grid=grid, plot_name=plot_name, export_folder=export_folder, plot_radius=plot_radius, max_occlusion=max_occlusion, sigma1=sigma1, min_pad_foliage=min_pad_foliage, max_pad_foliage=max_pad_foliage)

        return profile, summary


class BulkDensityProfileModel:
    def __init__(self, mass_ratio_profile:Union[np.array, None]=None, intercept:float=0, smoothing_factor:float=0.0):
        """
        Initialize object from mass ratio profile or initialize empty object
        Args:
            mass_ratio_profile: Leave None to initialize empty. Otherwise, must be array with two columns.
                First column must contain height in meters. Second column must contain mass ratio values.
                Height must be monotonically increasing (must represent an individual profile).
            intercept: model intercept
            smoothing_factor: Value used by voxelmon.utils.smooth() to smooth lidar values before prediction.
        """

        if mass_ratio_profile is not None:
            if mass_ratio_profile.shape[1] != 2:
                raise ValueError('mass_ratio_profile must be array with 2 columns (height and mass ratio value).')
            self.mass_ratio_profile = mass_ratio_profile
            self.update_interpolator()
        else:
            self.mass_ratio_profile = None

        self.smoothing_factor = smoothing_factor
        self.intercept = intercept

    @classmethod
    def from_file(cls, filepath):
        """
        Read all model information from a binary file produced by BulkDensityProfileModel.to_file()
        """

        import pickle
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj

    @classmethod
    def from_species_data(cls,
                         profile_data:pd.DataFrame,
                         species_cols:list[str],
                         height_col:str,
                         mass_ratio_dict:dict[float],
                         intercept:float=0,
                         smoothing_factor:float=0.0):
        """
        Initialize model given species proportions and mass ratio by species
        Args:
            profile_data: Dataframe containing species proportions at various heights along a **single** profile.
            species_cols: List of species group names. Each name must appear as separate column in table. Column values
                must represent species proportion of CBD. Sum of species proportions must be equal to 1.
            height_col: Name of column containing height in meters.
            mass_ratio_dict: Dictionary containing mass:lidar value ratio. Each col in species_cols must be key in dict.
            intercept: model intercept
            smoothing_factor: Value used by voxelmon.utils.smooth() to smooth lidar values before prediction.

        Returns: BulkDensityProfileModel
        """
        model =  cls()
        model.calculate_mass_ratio_profile(profile_data, species_cols,
                                          height_col, mass_ratio_dict,
                                          intercept, smoothing_factor)
        return model

    @classmethod
    def from_csv(cls, filepath, smoothing_factor=0):
        """Initialize model from templated CSV containing species proportions and mass ratio by species

        CSV must be formatted as follows:
        Row 0, col 0: intercept term
        Additional columns in row 0: mass ratio values for each species in model
        Row 1: Headers
        Col 0, additional rows: Bin height in meters
        Additional rows and columns: Proportion of CBD for this species and height
        """

        import numpy as np
        import pandas as pd
        params = pd.read_csv(filepath,header=None,nrows=1,index_col=None).to_numpy().flatten()
        df = pd.read_csv(filepath,skiprows=1,index_col=None)
        if len(params) != len(df.columns):
            raise ValueError('n_columns in row 0 did not match n_columns in table')

        if ~np.isfinite(params[0]):
            intercept = params[0]
        else:
            intercept = 0
        mass_ratio_dict = dict(zip(df.columns[1:], params[1:]))
        height_col = df.columns[0]
        model = cls()
        model.calculate_mass_ratio_profile(df, df.columns[1:], height_col,
                                           mass_ratio_dict, intercept, smoothing_factor)
        return model


    def update_interpolator(self):
        """Must update interpolator if self.mass_ratio_profile is modified"""
        from scipy import interpolate
        self.interpolator = interpolate.interp1d(x=self.mass_ratio_profile[:, 0],
                                                 y=self.mass_ratio_profile[:, 1],
                                                 bounds_error=False,
                                                 fill_value=(self.mass_ratio_profile[0, 1], # Fill with first & last val
                                                             self.mass_ratio_profile[-1, 1]),
                                                 assume_sorted=True)

    def calculate_mass_ratio_profile(self,
                                     profile_data:pd.DataFrame,
                                     species_cols:list[str],
                                     height_col:str,
                                     mass_ratio_dict:dict[float],
                                     intercept:float=0,
                                     smoothing_factor:float=0.0):
        """
        Calculate mass ratio profile given species proportions and mass ratio by species
        Args:
            profile_data: Dataframe containing species proportions at various heights along a **single** profile.
            species_cols: List of species group names. Each name must appear as separate column in table. Column values
                must represent species proportion of CBD. Sum of species proportions must be equal to 1.
            height_col: Name of column containing height in meters.
            mass_ratio_dict: Dictionary containing mass:lidar value ratio. Each col in species_cols must be key in dict.
            intercept: model intercept
            smoothing_factor: Value used by voxelmon.utils.smooth() to smooth lidar values before prediction.

        Returns:

        """

        self.mass_ratio_profile = np.zeros([profile_data.shape[0],2], np.float64)
        self.mass_ratio_profile[:,0] = profile_data[height_col]
        for species in species_cols:
            self.mass_ratio_profile[:,1] += profile_data[species] * mass_ratio_dict[species]
        self.intercept = intercept
        self.smoothing_factor = smoothing_factor
        self.update_interpolator()

        #TODO: Validate input data

    def predict(self,
                profile_data:pd.DataFrame,
                height_col:str,
                lidar_value_col:str,
                plot_id_col: str | None) -> np.ndarray:
        """
        Predict canopy bulk density profile given new lidar profile
        Args:
            profile_data: dataframe containing new data
            height_col: name of column containing height in meters
            lidar_value_col: name of column containing lidar values
            plot_id_col: name of column containing plot IDs. If None, assumes that there is only one profile.
            result_col: name of column to store results

        Returns: np.ndarray containing input data and prediction results.

        """
        from voxelmon import smooth
        #if self.intercept != 0:
        #    raise NotImplementedError('Non-zero intercept not currently implemented')

        if plot_id_col is not None:
            results = []
            # Filter to each plot and process independently
            for plot_id in profile_data[plot_id_col].unique():
                # Filter to plot
                df_filter = profile_data[profile_data[plot_id_col] == plot_id]
                # Smooth lidar values
                lidar_vals = smooth(df_filter[lidar_value_col], self.smoothing_factor).clip(min=0)
                # Interpolate mass ratio profile to match new heights
                mass_ratio_profile = self.interpolator(df_filter[height_col])
                results.append(lidar_vals * mass_ratio_profile + self.intercept)
            results = np.concatenate(results)
        else:
            # Smooth lidar values
            lidar_vals = smooth(profile_data[lidar_value_col], self.smoothing_factor).clip(min=0)
            # Interpolate mass ratio profile to match new heights
            mass_ratio_profile = self.interpolator(profile_data[height_col])
            results = lidar_vals * mass_ratio_profile + self.intercept

        return results

    def to_file(self,filepath):
        """
        Save all model information to a binary file.
        """

        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


class BulkDensityProfileModelFitter:
    def __init__(self,
                 profile_data: pd.DataFrame,
                 species_cols: list[str],
                 lidar_value_col: str,
                 cbd_col:str,
                 height_col: str,
                 plot_id_col: str,
                 class_id_col: str,
                 smoothing_factor: float = 0.0,
                 min_height: float = 1.0):
        """
        Initialize model fitter by providing profile data and column mapping
        Args:
            profile_data: Table containing estimates of CBD, species proportions, and lidar value
            species_cols: List of species group names. Each name must appear as separate column in table. Column values
                must represent species proportion of CBD. Sum of species proportions must be equal to 1.
            lidar_value_col: Name of column containing lidar values to be used in prediction (e.g. 'pad' or 'foliage')
            cbd_col: Name of column containing canopy bulk density values
            height_col: Name of column containing height labels. Heights must be in meters.
            plot_id_col: Name of column containing plot IDs. Plot IDs must separate individual profiles.
            class_id_col: Name of column containing class IDs. Data from all classes will be used to estimate mass
                ratio by species but species composition will be calculated separately for each class.
            smoothing_factor: Value used by voxelmon.utils.smooth() to smooth each profile's CBD and lidar values
            min_height: Minimum height considered in model fitting and prediction
        """

        self.species_cols = species_cols
        self.lidar_value_col = lidar_value_col
        self.cbd_col = cbd_col
        self.height_col = height_col
        self.plot_id_col = plot_id_col
        self.class_id_col = class_id_col
        self.smoothing_factor = smoothing_factor
        self.min_height = min_height
        self.mass_ratio_dict = None
        self.species_profiles = None

        # Smooth profiles (individually for each plot)
        from voxelmon import smooth
        profiles_list = []
        for plot in profile_data[plot_id_col].unique():
            profile_data_tmp = profile_data[(profile_data[plot_id_col] == plot)].copy()
            profile_data_tmp[lidar_value_col] = smooth(profile_data_tmp[lidar_value_col],
                                                       smoothing_factor=smoothing_factor).clip(min=0)
            profile_data_tmp[cbd_col] = smooth(profile_data_tmp[cbd_col], smoothing_factor=smoothing_factor).clip(min=0)
            profiles_list.append(profile_data_tmp)

        profile_data = pd.concat(profiles_list)
        profile_data = profile_data[profile_data[height_col] >= min_height]
        profile_data.loc[profile_data[cbd_col] < .00001, cbd_col] = 0
        self.profile_data = profile_data
        self.summarize_species_profiles()

        #TODO: Move additional preprocessing steps here

        #TODO: validate input data

    @classmethod
    def from_file(cls, filepath):
        """
        Read all model information from a binary file produced by BulkDensityProfileModelFitter.to_file()
        """

        import pickle
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def fit_mass_ratio_bayesian(self,
                                 prior_mean: np.ndarray,
                                 prior_std: np.ndarray,
                                 sigma_residuals:float = .02,
                                 sigma_intercept:float = .005,
                                 fit_intercept: bool = False,
                                 two_stage_fit: bool = False):
        """
        Fit self with Bayesian linear regression using prior coefficients and new observations

        Args:
            prior_mean: Prior estimates of the mass:lidar value coefficients (e.g. LMA estimates from previous studies).
            prior_std: Estimated standard deviation for prior coefficients. If uncertain, use large value
                representing weakly informative prior.
            fit_intercept: Use intercept in CBD prediction equation.
            two_stage_fit: Adjust mass ratio values to reduce bias in total canopy fuel load predictions

        Returns: None
        """
        import pymc as pm
        import arviz as az

        # Get species proportions
        X = self.profile_data[self.species_cols].to_numpy()

        # Scale lidar value by species proportions
        pad = self.profile_data[self.lidar_value_col].to_numpy().reshape(-1, 1)
        X *= pad

        y = self.profile_data[self.cbd_col].to_numpy()

        #if fit_intercept:
        #    raise NotImplementedError('Fit with intercept is not fully implemented')

        with pm.Model() as model:
            # Priors for coefficients
            betas = pm.Normal('betas', mu=prior_mean, sigma=prior_std, shape=(X.shape[1],))

            # Assume normal distribution of residuals
            sigma = pm.HalfNormal('sigma', sigma=sigma_residuals)

            if fit_intercept:
                # Prior for the intercept
                intercept = pm.Normal('intercept', mu=0, sigma=sigma_intercept)
                Y_obs = pm.Normal('Y_obs', mu=intercept + pm.math.dot(X, betas), sigma=sigma, observed=y)
            else:
                Y_obs = pm.Normal('Y_obs', mu=pm.math.dot(X, betas), sigma=sigma, observed=y)

            # Inference (sampling from the posterior)
            idata = pm.sample()

        # After sampling, you can inspect the posterior samples
        self.bayesian_samples = idata
        az.plot_trace(idata, combined=True)
        import matplotlib.pyplot as plt
        plt.show()
        self.fit_summary = az.summary(idata, round_to=2)
        if fit_intercept:
            self.fit_summary.index = ['intercept'] + self.species_cols + ['sigma']
            self.intercept = self.fit_summary['mean'].iloc[0]
            self.mass_ratio_dict = dict(zip(self.species_cols, self.fit_summary['mean'].iloc[1:len(self.species_cols) + 1]))
        else:
            self.fit_summary.index = self.species_cols + ['sigma']
            self.intercept = 0
            self.mass_ratio_dict = dict(zip(self.species_cols, self.fit_summary['mean'].iloc[:len(self.species_cols)]))
        #print(self.fit_summary)

        if two_stage_fit:
            import statsmodels.api as sm
            # Convert plot_id strings to vector of integers
            _, plot_id_arr = np.unique(self.profile_data[self.plot_id_col], return_inverse=True)
            # Get predicted cbd of each bin
            models = self.to_models()
            y_pred = np.zeros(y.shape, dtype=float)
            for veg_type in self.profile_data[self.class_id_col].unique():
                model = models[veg_type]
                class_mask = self.profile_data[self.class_id_col] == veg_type
                y_pred[class_mask] = model.predict(self.profile_data[class_mask],
                                                     self.height_col, self.lidar_value_col, self.plot_id_col)
            # Get sum of bins in each plot (pred and obs)
            pred_plot_sum = np.bincount(plot_id_arr, weights=y_pred)
            obs_plot_sum = np.bincount(plot_id_arr, weights=y)
            lm = sm.OLS(obs_plot_sum,pred_plot_sum).fit()
            self.adj_factor = lm.params[0]
            for species in self.mass_ratio_dict:
                self.mass_ratio_dict[species] *= self.adj_factor

        #TODO: Fix fit with intercept

        #TODO: Standardize outputs

        #TODO: Reimplement two-stage fit

    def summarize_species_profiles(self):
        """
        Summarize species distribution profile for each class
        Args:
            profile_data:
            species_cols:
            height_col:
            class_id_col:

        Returns: None. Data is written to self.species_profiles.
        """
        # Get species distribution profile for each class
        species_dist =self.profile_data[[self.class_id_col,self.height_col] + self.species_cols].groupby([self.class_id_col, self.height_col]).sum()
        speciesDistSum = species_dist.sum(axis=1)
        for col in species_dist.columns:
            species_dist[col] = species_dist[col] / speciesDistSum

        self.species_profiles = {}
        for classId in species_dist.index.get_level_values(0).unique():
            self.species_profiles[classId] = species_dist.loc[classId, :].reset_index().ffill()

    def to_models(self)->dict[BulkDensityProfileModel]:
        """Create BulkDensityProfileModel for each class in profile_data"""
        if self.mass_ratio_dict is None:
            raise ValueError('Mass ratio dictionary is None. Use fit_mass_ratio_bayesian()')
        if self.species_profiles is None:
            raise ValueError('Species profiles dictionary is None. Use summarize_species_profiles()')
        models = dict()
        for class_id in self.species_profiles.keys():
            species_profile = self.species_profiles[class_id]
            model = BulkDensityProfileModel.from_species_data(species_profile,self.species_cols,self.height_col,
                                                              self.mass_ratio_dict,self.intercept,self.smoothing_factor)
            models[class_id] = model

        return models



    def to_file(self, filepath):
        """
        Save all model information to a binary file.
        """

        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


