import pandas as pd
import time
import numpy as np
from pathlib import Path
import pickle
import os
import warnings
import matplotlib.pyplot as plt
from voxelmon import Grid,Pulses,PtxBlk360G1,get_files_list,plot_side_view,is_noise_ivf


gridRadius = 12 # Distance from grid center to edge
maxGridHeight = 30 # Height of grid above coordinate [0,0,0]
maxOcclusion = .75

inputFolder = 'G:/Scans/Fort Valley'
keyword = 'ptx'
exportFolder = 'D:/DataWork/WildBill/Post/'
modelPath = 'PPO.model'
process = True
generateFigures = True

with open(modelPath,'rb') as f:
    model = pickle.load(f)
sigma = model.sigma
feature = model.feature
cellSize = model.cellSize

baseExtents = [-gridRadius,-gridRadius,-gridRadius,gridRadius,gridRadius,maxGridHeight]

files = get_files_list(inputFolder, keyword)
start_time_all = time.time()
i=1

warnings.filterwarnings("ignore", category=RuntimeWarning)
#files = files[6:]
if process:
    if not Path(exportFolder).exists():
        os.makedirs(exportFolder)
    if not Path(exportFolder).joinpath('DEM').exists():
        os.mkdir(Path(exportFolder).joinpath('DEM'))
    if not Path(exportFolder).joinpath('Points').exists():
        os.mkdir(Path(exportFolder).joinpath('Points'))
    if not Path(exportFolder).joinpath('PAD').exists():
        os.mkdir(Path(exportFolder).joinpath('PAD'))
    if not Path(exportFolder).joinpath('PAD_Summary').exists():
        os.mkdir(Path(exportFolder).joinpath('PAD_Summary'))

    for ptxFile in files:
        start_time = time.time()
        print("Starting file ", i, " of ",len(files))
        baseFileName = os.path.splitext(os.path.basename(ptxFile))[0].split('-')[0]

        ptx = PtxBlk360G1(ptxFile,applyTranslation=False,applyRotation=False,dropNull=False)

        pulses = Pulses.from_point_cloud_array(ptx.xyz,ptx.origin)

        pulses_thin = pulses.crop(baseExtents)
        #z_range = pulses_thin.xyz[~is_noise_ivf(pulses_thin.xyz),2]
        #minHeight = z_range.min()
        #maxHeight = z_range.max()
        minHeight = pulses_thin.xyz[:,2].min()
        maxHeight = pulses_thin.xyz[:, 2].max()+1
        extents = baseExtents.copy()
        extents[2] = minHeight
        extents[5] = maxHeight

        #pulses_thin = pulses_thin.thin_distance_weighted_random(.25)
        grid = Grid(extents=extents,cellSize=cellSize)

        grid.create_dem_decreasing_window(pulses_thin)

        grid.calculate_pulse_metrics(pulses)

        #grid.calculate_eigenvalues(pulses_thin.array[:,3:6])

        grid.filter_pad_noise_ivf()
        grid.gaussian_filter_PAD()
        threshold = np.quantile(grid.pad[(grid.pad>0) & (grid.occlusion<=maxOcclusion)],.9)
        grid.classify_foliage_with_PAD(maxOcclusion=maxOcclusion,minPADFoliage=.01,maxPADFoliage=threshold)
        summary = grid.summarize_by_height(clipRadius=11.3)

        pulses_thin.to_csv(os.path.join(exportFolder,'Points/',baseFileName) + '.csv')

        grid.export_grid_as_csv(os.path.join(exportFolder,'PAD/',baseFileName) + '.csv')

        grid.export_dem_as_csv(os.path.join(exportFolder,'DEM/',baseFileName) + '.csv')

        summary.write_csv(os.path.join(exportFolder, 'PAD_Summary/', baseFileName) + '.csv')

        plt.plot(summary['foliage'], summary['height'])
        plt.show()

        print("Finished file ", i, " of ", len(files)," in ", round(time.time()-start_time,3)," seconds")
        i += 1

    print("Finished all files in ",round(time.time()-start_time_all)," seconds")

from scipy.ndimage import gaussian_filter1d
import math

tls_summary_files = get_files_list(exportFolder+'/PAD_Summary','.csv')
df_all = []
for file in tls_summary_files:
    df_tls = pd.read_csv(file)
    df_tls.insert(0,'Plot_ID',os.path.basename(file).strip('.csv'))
    df_tls[feature] = gaussian_filter1d(df_tls[feature], sigma=sigma)
    df_all.append(df_tls)
df_all = pd.concat(df_all)

df_all = df_all[df_all['height']>=1]
df_all = df_all.fillna(0)
df_all['CLASS'] = 'PPO'

results = model.predict(df_all)

plots = results['Plot_ID'].unique()

import matplotlib.pyplot as plt
results = results.fillna(0)


if generateFigures:
    import polars as pl
    for plotname in results['Plot_ID'].unique():
        summaryFilepath = [Path(filename) for filename in tls_summary_files if plotname in filename][0]
        basename = summaryFilepath.name
        dempath = Path('/'.join(list(summaryFilepath.parts[:-2])+['DEM',basename]))
        pointspath = Path('/'.join(list(summaryFilepath.parts[:-2])+['Points',basename]))
        pts = pl.read_csv(pointspath)
        demPts = pl.read_csv(dempath)
        results_filter = results[(results['Plot_ID']==plotname) & (results['height']>0)]
        f,[ax1,ax2] = plt.subplots(ncols=2,sharey=True,width_ratios=[2.5,1],figsize=[8,4])
        [arr,arr_extents] = plot_side_view(pts,direction=3,demPtsNormalize=demPts,returnData=True)
        ax1.imshow(arr,extent=arr_extents,aspect=2)
        ax2.plot(results_filter['biomassPred'],results_filter['height'],label='Lidar Estimate')
        ymax = max(ax1.get_ylim()[1],ax2.get_ylim()[1],14)
        ax1.set_ylim([0,ymax])
        ax2.set_ylim([0,ymax])
        ax2.set_yticks(ax1.get_yticks())
        ax2.legend()
        plt.savefig(exportFolder+basename+'.png')
        plt.show()