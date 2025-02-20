import pandas as pd
import numpy as np
import xarray as xr

files = pd.read_csv('G:/wscatclover/PrePostNames.csv')
i=0
for i in files.index:
    pre = pd.read_csv(files.loc[i,'pre'])

# Create example data
arr = np.random.rand(50, 50, 50)  # 3D numpy array representing 10cm voxels
dims = ("z", "y", "x")  # Naming the dimensions
coords = {"z": np.arange(50), "y": np.arange(50), "x": np.arange(50)}

# Convert to xarray DataArray
da = xr.DataArray(arr, dims=dims, coords=coords)

# Aggregate using coarsen
factor = 5  # 50cm / 10cm
aggregated_da = da.coarsen(z=factor, y=factor, x=factor, boundary="trim").mean()
