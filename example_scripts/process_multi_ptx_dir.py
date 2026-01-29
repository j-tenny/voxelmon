# Process plant area density for all ptx files in a directory. It is assumed that there are multiple plots which each
# have multiple scans that are already aligned. To set up, ensure that the center scan can be identified by a consistent
# suffix in the file name. Then ensure that all scans within the same plot contain the same plot_id in the file name, as
# defined by get_plot_id().

import pandas as pd
import time
import numpy as np
from pathlib import Path
import os
import warnings
import matplotlib.pyplot as plt

from voxelmon import TLS_PTX_Group,get_files_list,plot_side_view,directory_to_pandas

input_folder = r'C:\Users\john1\OneDrive - Northern Arizona University\Work\TontoNF\TLS\PTX\AllScans'
center_scan_key = '- Med Density 1.ptx'
results_summary_name = 'ResultsSummary.csv'
export_folder = r'D:\DataWork\TontoUpdatedResultsMulti'
process = True
generate_figures = True

plot_radius = 11.3 # Distance from grid center to edge
max_grid_height = 30 # Height of grid above coordinate [0,0,0]
max_occlusion = .8
cell_size = .1
min_height = 0.2

# Define a function to read the plot id from a scan filename
def get_plot_id(filename):
    return Path(filename).stem.split('-')[0]

### Setup file processing ###
# Get center scan for each plot
files_scan1 = get_files_list(input_folder, center_scan_key)
# Get list of all scans
files_all = get_files_list(input_folder, '.ptx')
# Segment a list of scans for each plot
files_grouped = []
for file_scan1 in files_scan1:
    plot_id = get_plot_id(file_scan1)
    files_grouped.append([file for file in files_all if plot_id in file])

start_time_all = time.time()
i = 1

warnings.filterwarnings("ignore", category=RuntimeWarning)

export_folder = Path(export_folder)

if process:
    for filegroup in files_grouped:

        start_time = time.time()
        print("Starting file ", i, " of ",len(files_grouped))

        plot_id = get_plot_id(filegroup[0])

        ptx = TLS_PTX_Group(filegroup)
        grid, profile, plot_summary = ptx.execute_default_processing(export_dir=export_folder, plot_name=plot_id, cell_size=cell_size,
                                                                     plot_radius=plot_radius, max_height=max_grid_height, max_occlusion=max_occlusion,
                                                                     sigma1=0, min_pad_foliage=.01, max_pad_foliage=6)
        profile['PLT_CN'] = plot_id

        print("Finished file ", i, " of ", len(files_grouped)," in ", round(time.time()-start_time,3)," seconds")
        i += 1

    print("Finished all files in ",round(time.time()-start_time_all)," seconds")

stop
profile_paths = directory_to_pandas(Path(export_folder) / 'PAD_Profile', filename_col='PLT_CN')
profiles = []

profiles = pd.concat(profiles)
profiles = profiles[profiles['HT']>=.2]

if generate_figures:
    import polars as pl
    for plotname in profiles['PLT_CN'].unique():
        #profile_path = [Path(filename) for filename in tls_summary_files if plotname in filename][0]
        dempath = Path('/'.join([str(export_folder), 'DEM', plotname + '.csv']))
        pointspath = Path('/'.join([str(export_folder), 'Points', plotname + '.csv']))
        profile = profiles[profiles['PLT_CN'] == plotname]
        pts = pl.read_csv(pointspath)
        demPts = pl.read_csv(dempath)
        f,[ax1,ax2] = plt.subplots(ncols=2,sharey=True,figsize=[8,4])
        [arr,arr_extents] = plot_side_view(pts,direction=3,demPtsNormalize=demPts,returnData=True)
        ax1.imshow(arr,extent=arr_extents,aspect=2)
        ax2.plot(profile['PAD'],profile['HT'],label='Plant area density (m^2/m^3)')
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
        plt.savefig(export_folder.joinpath(plotname + '.png'), dpi=300)
        plt.show()