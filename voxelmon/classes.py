import numpy as np
from numba import jit,njit,guvectorize,prange,float32,void,uint16,int64,uint32,int32,float64
import time

import voxelmon


class Grid:
    def __init__(self,extents,cellSize):
        self.cellSize = np.float64(cellSize)
        self._set_extents(extents)


        xCenters = np.arange(self.extents[0], self.extents[3], cellSize) + cellSize / 2
        yCenters = np.arange(self.extents[1], self.extents[4], cellSize) + cellSize / 2
        zCenters = np.arange(self.extents[2], self.extents[5], cellSize) + cellSize / 2

        self.shape = [len(xCenters), len(yCenters), len(zCenters)]

        z, y, x = np.meshgrid(zCenters, yCenters, xCenters, indexing='ij')

        self.centers = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T  # Grid centers increment along x, then y, then z
        self.centersXY = self.centers[:self.shape[0] * self.shape[1], :2]

        self.pDirected = np.zeros(self.shape, dtype=np.uint32)
        self.pTransmitted = np.zeros(self.shape, dtype=np.uint32)
        self.pIntercepted = np.zeros(self.shape, dtype=np.uint32)
        self.classification = np.zeros(self.shape, dtype=np.int8)

        self.classificationKey = {}

        self.geometricFeatures = np.zeros([self.centers.shape[0], 4], np.float32) - 50


    @classmethod
    def from_dims(cls,cellSize,gridHalfLengthX,gridHalfLengthY,gridHeight,gridBottom=None):
        # Define grid parameters
        if gridBottom is None:
            gridExtents = [-gridHalfLengthX, -gridHalfLengthY, min(-gridHalfLengthX, -gridHalfLengthY),
                            gridHalfLengthX,gridHalfLengthY,gridHeight] # xMin,yMin,zMin,xMax,yMax,zMax
        else:
            gridExtents = [-gridHalfLengthX, -gridHalfLengthY, gridBottom,
                           gridHalfLengthX, gridHalfLengthY, gridHeight]  # xMin,yMin,zMin,xMax,yMax,zMax
        return cls(gridExtents,cellSize)

    def _set_extents(self,extents):
        extents = np.array(extents).copy().astype(np.float64).round(5).flatten()
        ncells = (extents[3:] - extents[0:3]) // self.cellSize
        extents[3] = extents[0] + ncells[0] * self.cellSize
        extents[4] = extents[1] + ncells[1] * self.cellSize
        extents[5] = extents[2] + ncells[2] * self.cellSize
        self.extents = extents

    def calculate_eigenvalues(self,points):
        pointBins = (points // self.cellSize).astype(np.int32)
        #sortedIndices = pointBins.view([('x', np.float64), ('y', np.float64), ('z', np.float64)]).argsort(axis=0,order=['z','y','x']).flatten()
        #pointBins = pointBins[sortedIndices]
        #points = points[sortedIndices]
        occupiedBins, pointCount = np.unique(pointBins,axis=0,return_counts=True)
        occupiedBins = occupiedBins[pointCount>=3]

        # Create empty array to fill out
        gridBins = (self.centers // self.cellSize).astype(np.int32)

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

        calculate_eigenvalues_numba(points,pointBins,occupiedBins,binMin,binRange,self.geometricFeatures)

    def calculate_pulse_metrics(self,pulses, G = .5):

        @njit([void(float64[:, :], float64, float64[:], uint32[:, :, :], uint32[:, :, :], uint32[:, :, :])],parallel=True)
        def voxel_traversal(pulses, cellSize, gridExtents, pDirected, pTransmitted, pIntercepted):
            # @guvectorize([(float64[:,:],float64,float64[:], uint32[:,:,:], uint32[:,:,:],uint32[:,:,:],uint16)], '(a,b),(),(c),(d,e,f),(d,e,f),(d,e,f)->()', nopython=True, target='parallel')
            # def voxel_traversal(pulses,cellSize,gridExtents,pDirected,pTransmitted,pIntercepted,outputFlag):
            outputFlag = 1
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

                cellX = np.uint16((xstart - gridExtents[0]) // cellSize)
                cellY = np.uint16((ystart - gridExtents[1]) // cellSize)
                cellZ = np.uint16((zstart - gridExtents[2]) // cellSize)

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
                    pDirected[cellX, cellY, cellZ] += 1  # Pulse was directed at this voxel, may or may not have passed through

                    if (not intercepted) and (cellZ == cellZEnd) and (cellX == cellXEnd) and (cellY == cellYEnd):
                        pIntercepted[cellX, cellY, cellZ] += 1  # Pulse intercepted within this voxel
                        intercepted = True

                    if not intercepted:
                        pTransmitted[cellX, cellY, cellZ] += 1  # Pulse passed through this voxel

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

        voxel_traversal(pulses.array, self.cellSize, self.extents, self.pDirected, self.pTransmitted, self.pIntercepted)

        meanPathLength = .843*self.cellSize # From Grau et al 2017
        self.occlusion = 1 - (self.pIntercepted+self.pTransmitted) / self.pDirected
        self.occlusion[~np.isfinite(self.occlusion)] = 1
        self.pad = -np.log(1-(self.pIntercepted/(self.pIntercepted+self.pTransmitted))) / (G*meanPathLength)
        self.occlusion[~np.isfinite(self.pad)] = 1
        self.pad[~np.isfinite(self.pad)] = 0

    def add_pulse_metrics(self,grid, G = .5):
        # Combine pulse metrics from two Grid objects. Grids must overlap exactly (same extents, same cell size).
        self.pDirected += grid.pDirected
        self.pIntercepted += grid.pIntercepted
        self.pTransmitted += grid.pTransmitted
        meanPathLength = .843 * self.cellSize  # From Grau et al 2017
        self.pad = -np.log(1-(self.pIntercepted/(self.pIntercepted+self.pTransmitted))) / (G*meanPathLength)
        self.occlusion = np.minimum(self.occlusion,grid.occlusion)
        self.occlusion[~np.isfinite(self.occlusion)] = 1
        self.occlusion[~np.isfinite(self.pad)] = 1
        self.pad[~np.isfinite(self.pad)] = 0



    def filter_pad_noise_ivf(self,windowRadius=15,minCountPresent=5):
        # Search a moving window around each grid cell, set PAD to 0 if cell has few neighbors
        # Window size is total width of window with cells as the unit. Must be odd number?
        from scipy import ndimage
        presence = (self.pad>0).astype(np.float32)
        count = ndimage.uniform_filter(presence,windowRadius) * windowRadius**2
        self.pad[count<minCountPresent] = 0


    def create_dem_decreasing_window(self, pulses, windowSizes = [5,2.5,1], heightThresholds=[1,.5,.25]):
        import polars as pl
        import scipy

        try:
            points = pulses.xyz
        except:
            points = pulses[:,0:3]

        # Get lowest point in base grid
        grid = self.bin2D(points,pl.min('z'))
        grid[np.isnan(grid)]=99
        centers = self.centersXY
        # Create a grid of coordinates corresponding to the array indices
        x, y = np.indices(grid.shape)

        points = pl.DataFrame({'x':points[:,0],'y':points[:,1],'z':points[:,2],'ground':-2})
        points = points.with_columns(pl.col('x').floordiv(self.cellSize).cast(pl.Int32).alias('xBin'),
                                     pl.col('y').floordiv(self.cellSize).cast(pl.Int32).alias('yBin'))

        # Get lowest point in decreasing windows
        for windowSize,heightThresh in zip(windowSizes,heightThresholds):
            # Get low points in moving window
            cellRadius = round(windowSize/2/self.cellSize)
            windowShape = [cellRadius*2+1,cellRadius*2+1]
            grid = scipy.ndimage.percentile_filter(grid,.25,windowShape,mode='nearest')

            # Interpolate remaining missing values
            maskMissing = grid == 99
            pointsValid = np.array((x[~maskMissing], y[~maskMissing])).T
            valuesValid = grid[~maskMissing]
            pointsMissing = np.array((x[maskMissing], y[maskMissing])).T
            grid[maskMissing] = scipy.interpolate.griddata(pointsValid, valuesValid, pointsMissing, method='linear')

            # Relate elevation data to point cloud
            dem_df = pl.DataFrame({'xBin':(centers[:,0]//self.cellSize).astype(np.int32),
                                   'yBin':(centers[:,1]//self.cellSize).astype(np.int32),
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
        grid[maskMissing] = scipy.interpolate.griddata(pointsValid, valuesValid, pointsMissing, method='linear')

        self.dem = grid
        self.hag = self.centers[:,2] - np.tile(grid.flatten(),self.shape[2])

    def classify_foliage_with_PAD(self,maxOcclusion=.75,minPADFoliage=.05,maxPADFoliage=6):
        self.classification[:,:,:] = -2
        self.classification[self.occlusion>maxOcclusion] = -1
        self.classification[self.pad<0] = -1
        self.classification[self.pad>=minPADFoliage] = 3
        self.classification[self.pad>maxPADFoliage] = 5
        self.classificationKey = {-2:'empty',-1:'occluded',3:'foliage',5:'nonfoliage'}

    def gaussian_filter_PAD(self,sigma=.1,maxOcclusion=.75):

        from scipy import ndimage,interpolate
        # Interpolate missing values
        #occlusionFlat = self.occlusion.flatten()
        #pointsValid = self.centers[occlusionFlat<=maxOcclusion]
        #valuesValid = occlusionFlat[occlusionFlat<=maxOcclusion]
        #pointsMissing = self.centers[occlusionFlat>maxOcclusion]
        #vals = interpolate.griddata(pointsValid, valuesValid, pointsMissing, method='linear')
        #self.pad[self.occlusion>maxOcclusion] = vals.reshape(self.shape)

        self.pad = ndimage.gaussian_filter(self.pad,sigma)


    def bin2D(self,pulses,function):
        # Function should be from polars and should specify a column name x, y, or z, e.g. pl.min('z')
        import polars as pl
        try:
            points_df = pl.DataFrame({'x':pulses.xyz[:,0],'y':pulses.xyz[:,1],'z':pulses.xyz[:,2]})
        except:
            points_df = pl.DataFrame({'x':pulses[:,0],'y':pulses[:,1],'z':pulses[:,2]})

        points_df = points_df.with_columns(pl.col('x').floordiv(self.cellSize).cast(pl.Int32).alias('xBin'),
                                           pl.col('y').floordiv(self.cellSize).cast(pl.Int32).alias('yBin'))

        bins_df = pl.DataFrame({'xBin': (self.centersXY[:,0]//self.cellSize).astype(np.int32),
                                'yBin': (self.centersXY[:,1]//self.cellSize).astype(np.int32)})

        binVals = points_df.drop_nulls().group_by(['xBin','yBin']).agg(function)

        return bins_df.join(binVals,['xBin','yBin'],'left').to_numpy()[:,2].reshape([self.shape[0],self.shape[1]],order='c')

    def to_polars(self):
        import polars
        df = polars.DataFrame({'x': self.centers[:, 0], 'y': self.centers[:, 1], 'z': self.centers[:, 2], 'hag':self.hag,
                                'pDirected': self.pDirected.flatten('F'),'pTransmitted': self.pTransmitted.flatten('F'),
                                'pIntercepted': self.pIntercepted.flatten('F'),'occlusion': self.occlusion.flatten('F'),
                                'pad': self.pad.flatten('F'), 'linearity':self.geometricFeatures[:,0],
                                'planarity':self.geometricFeatures[:,1],'eigenentropy':self.geometricFeatures[:,2],
                                'verticality':self.geometricFeatures[:,3],'classification':self.classification.flatten('F')})
        return df


    def summarize_by_height(self,clipRadius = None, foliageClasses = [3],emptyClass=-2):
        import polars as pl

        df = self.to_polars()

        if clipRadius!=None:
            df = df.filter(((pl.col('x')**2 + pl.col('y')**2)**.5)<clipRadius)

        # Filter bad height data
        df = df.filter((pl.col('hag')>=0) & (pl.col('hag').is_not_nan()))

        # Assign height bins
        df = df.with_columns(pl.col('hag').floordiv(self.cellSize).cast(pl.Int32).alias('heightBin'))

        # Count voxels by class and height bin
        summary = df.pivot(on="classification", index="heightBin", values='classification', aggregate_function=pl.len())
        summary = summary.fill_null(0)

        # Assign meaningful column names
        summary.columns =['heightBin'] + [self.classificationKey[int(col)] for col in summary.columns[1:]]

        if 'occluded' not in summary.columns:
            summary = summary.with_columns(pl.lit(0.0).alias('occluded'))

        # Convert count to proportion of non-occluded volume
        volNO = summary.drop(['heightBin','occluded']).sum_horizontal()
        volTotal = summary.drop(['heightBin']).sum_horizontal()
        pctOccluded = summary['occluded'] / volTotal
        summary = summary.with_columns(summary.drop(['heightBin']) / volNO)
        summary = summary.with_columns(pctOccluded)


        # Get mean PAD within foliage and empty space, ignoring occluded areas and other classes
        pad = df.filter(pl.col('classification').is_in(foliageClasses+[emptyClass]))
        pad = pad.group_by('heightBin').agg(pl.mean('pad')).sort('heightBin')

        summary = summary.join(pad,on='heightBin',how='full',coalesce=True).sort('heightBin')

        # Clean up
        summary = pl.DataFrame((summary['heightBin'].cast(pl.Float64) * (self.cellSize)).alias('height')).hstack(summary)
        summary = summary.fill_null(0).fill_nan(0)

        return summary

    def export_height_summary_as_csv(self,filepath,clipRadius = None,foliageClasses = [3],emptyClass=-2):
        df = self.summarize_by_height(clipRadius = clipRadius,foliageClasses=foliageClasses,emptyClass=emptyClass)
        df.write_csv(filepath)

    def export_grid_as_csv(self, filepath):
        df = self.to_polars()
        df.write_csv(filepath)

    def export_dem_as_csv(self,filepath):
        import polars
        df = polars.DataFrame({'x':self.centersXY[:,0],'y':self.centersXY[:,1],'z':self.dem.flatten()})
        df.write_csv(filepath)






# Read and format pulses array

class Pulses:
    def __init__(self,pulses_array):
        # Pulses array must be a 2D numpy array with type float64 with columns ordered
        # 0:x_origin, 1:y_origin, 2:z_origin, 3:x_end, 4:y_end, 5:z_end, 6:unit_x, 7:unit_y, 8:unit_z, 9:total_length
        if pulses_array.shape[1] != 10:
            raise IndexError('pulses_array must have 10 correctly formatted columns, otherwise use Pulses.from_point_cloud_array()')
        self.array = pulses_array
        self.xyz = self.array[:, 3:6]

    @classmethod
    def from_point_cloud_array(cls,xyzArray,origin):
        # xyzArray must be a numpy array with column order x, y, z
        xyzArray = xyzArray.astype(np.float64)
        self = cls.initialize_empty_array(nRecords=xyzArray.shape[0])

        self.array[:, 0] = origin[0]
        self.array[:, 1] = origin[1]
        self.array[:, 2] = origin[2]

        self.array[:, 3:6] = xyzArray[:, 0:3]

        self.calculate_unit_vectors()

        return self


    @classmethod
    def from_las(cls,las_path,origin):
        import laspy
        las = laspy.read(las_path)
        return cls.from_point_cloud_array(las.xyz,origin)

    @classmethod
    def initialize_empty_array(cls,nRecords):
        nCols = 10 # x_origin, y_origin, z_origin, x_end, y_end, z_end, unit_x, unit_y, unit_z, total_length
        return cls(np.zeros([nRecords, nCols], np.float64))

    def calculate_unit_vectors(self):
        diff = self.array[:, 3:6] - self.array[:, 0:3]
        self.array[:, 9] = (diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2) ** .5  # calculate total length
        self.array[:, 6] = diff[:, 0] / self.array[:, 9]  # calculate x component of unit vector
        self.array[:, 7] = diff[:, 1] / self.array[:, 9]  # calculate y component of unit vector
        self.array[:, 8] = diff[:, 2] / self.array[:, 9]  # calculate z component of unit vector

    def crop(self,extents):
        cropped = self.array[(self.array[:,3]>=extents[0]) & (self.array[:,4]>=extents[1]) & \
                            (self.array[:,5]>=extents[2]) & (self.array[:,3]<=extents[3]) & \
                            (self.array[:,4]<=extents[4]) & (self.array[:,5]<=extents[5])]
        return Pulses(cropped)

    def thin_distance_weighted_random(self,propRemaining):
        distsq = self.array[:,9]**2
        weights = distsq / distsq.sum()
        randIndex = np.random.choice(distsq.size,int(distsq.size * propRemaining),replace=False,p=weights)
        thinned_array = self.array[randIndex]
        return Pulses(thinned_array)


    def to_polars(self):
        import polars
        df = polars.DataFrame({'x':self.array[:,3],'y':self.array[:,4],'z':self.array[:,5],
                               'x_origin':self.array[:,0],'y_origin':self.array[:,1],'z_origin':self.array[:,2],
                               'x_unit':self.array[:,6],'y_unit':self.array[:,7],'z_unit':self.array[:,8],
                               'length':self.array[:,9]})
        return df

    def to_csv(self,filepath,coordinatesOnly=True):
        df=self.to_polars()
        if coordinatesOnly:
            import polars as pl
            df = df.select(pl.col('x'),pl.col('y'),pl.col('z'))
        df.write_csv(filepath)


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


class FoliageProfileModel:
    def __init__(self):
        pass

    def fit(self, lidarProfile, biomassProfile, biomassCols, cellSize, sigma, lidarValueCol,fitIntercept=False,twoStageFit=False, plotIdCol='Plot_ID', heightCol='height', classIdCol='CLASS'):
        import statsmodels.formula.api as smf
        from patsy.contrasts import Sum

        from patsy.contrasts import ContrastMatrix

        def _name_levels(prefix, levels):
            return ["[%s%s]" % (prefix, level) for level in levels]

        class Simple(object):
            def _simple_contrast(self, levels):
                nlevels = len(levels)
                contr = -1.0 / nlevels * np.ones((nlevels, nlevels - 1))
                contr[1:][np.diag_indices(nlevels - 1)] = (nlevels - 1.0) / nlevels
                return contr

            def code_with_intercept(self, levels):
                contrast = np.column_stack(
                    (np.ones(len(levels)), self._simple_contrast(levels))
                )
                return ContrastMatrix(contrast, _name_levels("Simp.", levels))

            def code_without_intercept(self, levels):
                contrast = self._simple_contrast(levels)
                return ContrastMatrix(contrast, _name_levels("Simp.", levels[:-1]))


        self.heightCol = heightCol
        self.cellSize = cellSize
        self.sigma = sigma
        self.feature = lidarValueCol

        weights = biomassProfile.join(biomassProfile[plotIdCol].value_counts(),on=plotIdCol)['count']
        weights = weights / weights.sum()

        # Use linear modelling to get feature:biomass coefficient for each species
        if fitIntercept:
            lm = smf.mixedlm(formula=lidarValueCol + '~' + '+'.join(biomassCols),
                             data=biomassProfile, groups=biomassProfile[plotIdCol],
                             re_formula="0+TOTAL").fit()
            #lm = smf.wls(lidarValueCol + '~' + '+'.join(biomassCols), biomassProfile, weights=1.0).fit()
            coef = lm.params.copy()
            self.intercept = coef.iloc[0]
            coef[coef == 0] = -1
            coef = 1 / coef
            coef[coef <= 0] = 0
            self.bulkDensity = coef.iloc[1:len(biomassCols)+1].to_dict()
        else:
            lm = smf.mixedlm(formula= lidarValueCol + '~' + '+'.join(biomassCols)+'-1',
                             data = biomassProfile, groups = biomassProfile[plotIdCol],
                             re_formula="~0+TOTAL").fit()
            #lm = smf.wls(lidarValueCol + '~' + '+'.join(biomassCols)+f"+ TOTAL:C({plotIdCol},Simple) -1",biomassProfile,weights=weights).fit()
            #lm = smf.wls(lidarValueCol + '~' + '+'.join(biomassCols) + "-1",biomassProfile, weights=weights).fit()

            coef = lm.params.copy()
            #coef += coef.iloc[len(biomassCols)] # Mean effect of total
            coef[coef==0]=-1
            coef = 1 / coef
            coef[coef<=0] = 0
            self.bulkDensity = coef.iloc[:len(biomassCols)].to_dict()
            self.intercept = 0


        # Get species distribution profile for each class
        speciesDist = biomassProfile[[classIdCol,heightCol]+biomassCols].groupby([classIdCol,heightCol]).sum()
        speciesDistSum = speciesDist.sum(axis=1)
        for col in speciesDist.columns:
            speciesDist[col]=speciesDist[col] / speciesDistSum

        self.speciesProfiles = {}
        for classId in biomassProfile[classIdCol].unique():
            self.speciesProfiles[classId] = speciesDist.loc[classId,:].reset_index().ffill()

        if twoStageFit:
            # Correct for biased predictions
            predicted = self.predict(lidarProfile=lidarProfile,lidarValueCol=lidarValueCol,heightCol=heightCol,classIdCol=classIdCol,resultCol='biomassPred')
            predicted_plot = predicted.pivot_table(values=['TOTAL','biomassPred'],index=plotIdCol,aggfunc='sum')
            lm2 = smf.ols('TOTAL~biomassPred-1',predicted_plot).fit()
            correction = float(lm2.params.iloc[0])
            for key in self.bulkDensity.keys():
                self.bulkDensity[key] *= correction

    def predict(self,lidarProfile,lidarValueCol='pad',heightCol='height',classIdCol='CLASS',resultCol='biomassPred'):
        import pandas as pd
        results = []
        for classId in lidarProfile[classIdCol].unique():
            df_filter = lidarProfile[lidarProfile[classIdCol]==classId].copy()
            df_filter = df_filter.drop(columns=self.bulkDensity.keys(),errors='ignore')
            df_filter = df_filter.merge(self.speciesProfiles[classId],left_on=heightCol,right_on=self.heightCol,how='left').ffill()
            result = pd.Series(np.zeros(df_filter.shape[0],np.float64))
            for species in self.bulkDensity.keys():
                result = result + (df_filter[lidarValueCol] - self.intercept) * df_filter[species] * self.bulkDensity[species]
            df_filter[resultCol] = result
            results.append(df_filter)
        results = pd.concat(results)

        return results





