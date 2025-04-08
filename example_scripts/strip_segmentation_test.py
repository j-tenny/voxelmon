import pandas as pd
import polars as pl
import voxelmon
from skimage import segmentation, feature, exposure, restoration, measure, morphology
from scipy.ndimage import median_filter,gaussian_filter,gaussian_filter1d
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import plotly.graph_objects as go

df = pd.read_csv('../test_outputs/Segmentation_test_strip.csv',index_col=[0,1,2],usecols=['//X','Y','Z','hag','pad'])
ds = xr.Dataset.from_dataframe(df)

points = pl.read_csv('../test_outputs/Segmentation_test_strip_points.csv',)


pad = ds['pad'].to_numpy()
hag_mask = (ds['hag']>=1).to_numpy()
empty_mask = pad <= .01
#pad[hag_mask] = np.ma.masked_array(pad,hag_mask)
#pad[~hag_mask] = np.nan

for i in range(pad.shape[2]):
    pad_slice = pad[:,:,i]
    pad[:,:,i] = restoration.denoise_bilateral(pad_slice,win_size=5,sigma_color=1,sigma_spatial=.5)

#pad = np.ma.masked_array(pad,hag_mask)
#pad[~hag_mask] = np.nan

pad = gaussian_filter1d(pad,2,axis=2)
#pad = np.ma.masked_array(pad,hag_mask)
#pad[~hag_mask] = np.nan
#pad = exposure.rescale_intensity(pad,out_range=(0,1))
pad = exposure.equalize_adapthist(pad,9)

# Identify blobs
#blobs=feature.blob_dog(pad,min_sigma=.25,max_sigma=50,threshold=.05,threshold_rel=None,overlap=.3)
blobs=feature.peak_local_max(pad,min_distance=2,threshold_abs=.05)
#sigma = blobs[:,3].mean()
#print(sigma)
sigma = 1.5

# Transform seed locations to 3D image
blobs = np.round(blobs).astype(int)
x, y, z  = blobs[:, 0], blobs[:, 1], blobs[:,2]
seeds = np.zeros(pad.shape, dtype=int)
# Assign unique labels for seeds starting from 1 (0 is unclassified)
labels = np.arange(1, len(y) + 1)
seeds[x,y,z] = labels
# Fill empty space and ground with -1
seeds = segmentation.expand_labels(seeds,1)
seeds[empty_mask] = -1
seeds[~hag_mask] = -1

print(f'n blobs: {len(np.unique(seeds))}')

# Segment from initial seed points
pad = gaussian_filter(pad,sigma)
#regions = segmentation.watershed(-pad,seeds)
regions = segmentation.random_walker(pad,seeds,beta=130)

# Reclass unclassified points
regions[regions<0] = 0
# Unclassify small regions
#regions = morphology.remove_small_objects(regions,min_size=10)
# Reset null space
regions[empty_mask] = -1
regions[~hag_mask] = -1

# Expand existing regions
for id in np.unique(regions[regions>0]):
    seed_point = tuple(np.argwhere(regions==id)[0,:])
    regions_temp = regions.copy()
    regions_temp[regions==id] = 0
    mask = segmentation.flood(regions_temp,seed_point)
    regions[mask] = id


print(f'n blobs: {len(np.unique(regions))}')

df['cluster'] = regions.ravel('f')
df.to_csv('../test_outputs/Segmentation_test_strip.csv')