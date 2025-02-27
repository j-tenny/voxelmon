import pandas as pd
import time
import numpy as np
from pathlib import Path
import pickle
import os
import warnings
import matplotlib.pyplot as plt

import voxelmon.utils
from voxelmon import Grid,Pulses,PtxBlk360G1,get_files_list,plot_side_view, BulkDensityProfileModel
import pyrothermel

input_folder = r'D:\DataWork\SanCarlos\SanCarlosFeb25\PTX'
keyword = '.ptx'
field_summary_name = 'FieldSummary.csv'
results_summary_name = 'ResultsSummary.csv'
exportFolder = 'G:/wscatclover/'
canopy_model_path = 'G:/wscatclover/wscatclover_species_composition.csv'
cbd_axis_limit = .08
process = False
generateFigures = True

plotRadius = 11.3 # Distance from grid center to edge
maxGridHeight = 30 # Height of grid above coordinate [0,0,0]
maxOcclusion = .75
cellSize = .1
min_height = 1

canopyModel = BulkDensityProfileModel.from_csv(canopy_model_path)
field_summary = pd.read_csv(Path(input_folder).joinpath(field_summary_name), index_col='EVENT_ID')

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

        ptx = PtxBlk360G1(ptxFile,applyTranslation=True,applyRotation=True,dropNull=False)
        profile, plot_summary = ptx.execute_default_processing(export_folder=exportFolder, plot_name=baseFileName, cell_size=cellSize,
                                                         plot_radius=plotRadius, max_height=maxGridHeight, max_occlusion=maxOcclusion,
                                                         sigma1=0, min_pad_foliage=.01, max_pad_foliage=6)
        profile['EVENT_ID'] = baseFileName
        profile['CANOPY_CLASS'] = field_summary.loc[baseFileName,'CANOPY_CLASS']
        profile['CBD'] = canopyModel.predict(profile, lidar_value_col='pad', height_col='height', plot_id_col='EVENT_ID')

        profile.to_csv(exportFolder.joinpath(baseFileName+'.csv'),index=False)

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
summary = voxelmon.utils.summarize_profiles(profiles,bin_height=cellSize,min_height=min_height,fsg_threshold=.011,cbd_col='CBD',height_col='height',pad_col='pad',occlusion_col='occluded',plot_id_col='EVENT_ID')
summary = summary.set_index('EVENT_ID')
summary = summary.join(field_summary,how='inner')
summary['LOAD_TOTAL'] = summary['LOAD_FDF']+summary['LOAD_10HR']+summary['LOAD_100HR']+summary['LOAD_LH']+summary['LOAD_LW']

behave_results = []
for plotname in summary.index:
    fm = pyrothermel.FuelModel.from_existing(summary.loc[plotname,'SURFACE_CLASS'])
    fm.fuel_bed_depth = summary.loc[plotname,'FUELBED_HT']/100
    fm.fuel_load_one_hour = summary.loc[plotname,'LOAD_FDF']
    fm.fuel_load_ten_hour = summary.loc[plotname,'LOAD_10HR']
    fm.fuel_load_hundred_hour = summary.loc[plotname, 'LOAD_100HR']
    fm.fuel_load_live_herbaceous = summary.loc[plotname, 'LOAD_LH']
    fm.fuel_load_live_woody = summary.loc[plotname, 'LOAD_LW']
    ms = pyrothermel.MoistureScenario.from_existing(1,2)
    run = pyrothermel.PyrothermelRun(fm,ms,40,canopy_base_height=summary.loc[plotname,'fsg'],canopy_bulk_density=summary.loc[plotname,'cbd_max'])
    run.run_surface_fire_in_direction_of_max_spread()
    result = run.run_crown_fire_scott_and_reinhardt()
    result['torching_index'] = run.calculate_torching_index(max_wind_speed=1000)
    result['crowning_index'] = run.calculate_crowning_index(max_wind_speed=1000)
    result['EVENT_ID'] = plotname
    behave_results.append(result)
behave_results = pd.DataFrame(behave_results)
behave_results = behave_results.set_index('EVENT_ID')

summary = summary.join(behave_results,how='inner')

summary.to_csv(exportFolder.joinpath(results_summary_name))


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
        ax2.axhspan(summary.loc[plotname,'fsg_h1'],summary.loc[plotname,'fsg_h2'],color='yellow', alpha=0.3, label='Fuel Strata Gap (FSG)')
        ax2.plot(profile['CBD'],profile['height'],label='Canopy Bulk Density (kg/m^3)')
        ax2.axvline(summary.loc[plotname,'cbd_max'],linestyle='--',color='black',label='Max Canopy Bulk Density')
        ax2.axvline(.011, linestyle='--', color='yellow', label='FSG Cutoff')
        ymax = max(ax1.get_ylim()[1],ax2.get_ylim()[1],14)
        ax1.set_ylim([0,ymax])
        ax2.set_ylim([0,ymax])
        if cbd_axis_limit is not None:
            ax2.set_xlim([0,cbd_axis_limit])
        ax2.set_yticks(ax1.get_yticks())
        ax1.text(0,1.1,plotname, transform=ax1.transAxes, fontsize=12, ha='left')
        ax1.set_ylabel('Height (m)')
        ax1.set_xlabel('Easting (m)')
        ax2.set_xlabel('Canopy Bulk Density (kg/m^3)')
        ax2.legend(loc="upper right", prop={'size': 'small'})
        table_data = [['Max CBD','Fuel Strata Gap', 'Surface Fuel Load', 'Fuelbed Depth', 'Spread Rate', 'Intensity', 'Flame Length', 'Torching Index', 'Crowning Index'],
                      [summary.loc[plotname,'cbd_max'].round(4),summary.loc[plotname,'fsg'].round(1),summary.loc[plotname,'LOAD_TOTAL'].round(2),summary.loc[plotname,'FUELBED_HT'].round(1),summary.loc[plotname,'spread_rate'].round(3),summary.loc[plotname,'fireline_intensity'].round(1),summary.loc[plotname,'flame_length'].round(1),summary.loc[plotname,'torching_index'],summary.loc[plotname,'crowning_index']],
                      ['kg/m^3','m','kg/m^2','cm','km/hr','kW/m','m','km/hr','km/hr']]
        table_data = np.array(table_data).T
        ax2.table(cellText=table_data, colLabels=['Name', 'Value','Units'], cellLoc='center', bbox=[1.1, 0, .75, 1],colWidths=[.5,.25,.25])
        ax2.text(1.1, -.1, "Potential fire behavior based on \n40km/hr wind; 'very low' moisture", transform=ax2.transAxes, fontsize=8, ha='left')
        f.tight_layout(pad=2)
        plt.savefig(exportFolder.joinpath(plotname+'.png'), dpi=300)
        plt.show()