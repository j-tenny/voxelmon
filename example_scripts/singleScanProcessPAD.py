import pandas as pd
import time
import numpy as np
from pathlib import Path
import os
import warnings
import matplotlib.pyplot as plt

from voxelmon import TLS_PTX,get_files_list,plot_side_view

input_folder = 'G:/Yavapai2_east/TLS/PTX/'
keyword = '.ptx'
results_summary_name = 'ResultsSummary.csv'
exportFolder = 'G:/Yavapai2_east/TLS/'
process = True
generateFigures = True

plotRadius = 11.3 # Distance from grid center to edge
maxGridHeight = 30 # Height of grid above coordinate [0,0,0]
maxOcclusion = .8
cellSize = .1
min_height = 0.2

files = get_files_list(input_folder, keyword, recursive=False)
start_time_all = time.time()
i=1

warnings.filterwarnings("ignore", category=RuntimeWarning)

exportFolder = Path(exportFolder)

if process:
    for ptxFile in files:
        start_time = time.time()
        print("Starting file ", i, " of ",len(files))
        baseFileName = os.path.splitext(os.path.basename(ptxFile))[0].split('-')[0]

        ptx = TLS_PTX(ptxFile, apply_translation=True, apply_rotation=True, drop_null=False)
        grid, profile, plot_summary = ptx.execute_default_processing(export_dir=exportFolder, plot_name=baseFileName, cell_size=cellSize,
                                                                     plot_radius=plotRadius, max_height=maxGridHeight, max_occlusion=maxOcclusion,
                                                                     sigma1=0, min_pad_foliage=.01, max_pad_foliage=6)
        profile['EVENT_ID'] = baseFileName

        print("Finished file ", i, " of ", len(files)," in ", round(time.time()-start_time,3)," seconds")
        i += 1

    print("Finished all files in ",round(time.time()-start_time_all)," seconds")

profile_paths = get_files_list(exportFolder, '.csv',recursive=False)
profiles = []
for profile_path in profile_paths:
    if results_summary_name in profile_path:
        continue
    else:
        profile = pd.read_csv(profile_path)
        profiles.append(profile)
profiles = pd.concat(profiles)
profiles = profiles[profiles['height']>=.2]

if generateFigures:
    import polars as pl
    for plotname in profiles['EVENT_ID'].unique():
        #profile_path = [Path(filename) for filename in tls_summary_files if plotname in filename][0]
        dempath = Path('/'.join([str(exportFolder),'DEM',plotname+'.csv']))
        pointspath = Path('/'.join([str(exportFolder),'Points',plotname+'.csv']))
        profile = profiles[profiles['EVENT_ID'] == plotname]
        pts = pl.read_csv(pointspath)
        demPts = pl.read_csv(dempath)
        f,[ax1,ax2] = plt.subplots(ncols=2,sharey=True,figsize=[8,4])
        [arr,arr_extents] = plot_side_view(pts,direction=3,demPtsNormalize=demPts,returnData=True)
        ax1.imshow(arr,extent=arr_extents,aspect=2)
        ax2.plot(profile['pad'],profile['height'],label='Plant area density (m^2/m^3)')
        ymax = max(ax1.get_ylim()[1],ax2.get_ylim()[1],14)
        ax1.set_ylim([0,ymax])
        ax2.set_ylim([0,ymax])
        ax2.set_yticks(ax1.get_yticks())
        ax1.text(0,1.1,plotname, transform=ax1.transAxes, fontsize=12, ha='left')
        ax1.set_ylabel('Height (m)')
        ax1.set_xlabel('Easting (m)')
        ax2.set_xlabel('Plant area density (m^2/m^3)')
        ax2.legend(loc="upper right", prop={'size': 'small'})
        f.tight_layout(pad=2)
        plt.savefig(exportFolder.joinpath(plotname+'.png'), dpi=300)
        plt.show()