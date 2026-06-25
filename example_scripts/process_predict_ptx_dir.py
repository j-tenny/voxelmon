# Process plant area density and predict canopy bulk density and potential fire behavior by applying existing model
# Fire behavior estimates require pyrothermel `pip install pyrothermel`
import pandas as pd
import time
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import voxelmon.utils
from voxelmon import TLS_PTX,get_files_list,plot_side_view, BulkDensityProfileModel
import pyrothermel
import argparse
import os

def process(input_folder, export_folder, field_summary_path, canopy_model_path,
            cell_size, plot_radius, min_height, max_grid_height, max_occlusion,
            cbd_axis_limit, wind_speed, wind_direction, do_processing,
            generate_figures):
    # RUN ##########################################################
    # Initialize canopy model from csv
    canopy_model = BulkDensityProfileModel.from_csv(canopy_model_path)
    # Read field data
    field_summary = pd.read_csv(Path(input_folder).joinpath(field_summary_path), index_col='PLT_CN')
    # Get ptx filepaths
    files = get_files_list(input_folder, '.ptx', recursive=False)

    # Run processing on each ptx file
    start_time_all = time.time()
    i = 1
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    export_folder = Path(export_folder)

    if do_processing:
        for ptx_file in files:
            start_time = time.time()
            print("Starting pre-processing for file ", i, " of ", len(files))
            base_file_name = os.path.splitext(os.path.basename(ptx_file))[0].split('-')[0]

            ptx = TLS_PTX(ptx_file, apply_translation=False, apply_rotation=True, drop_null=False)
            grid, profile, plot_summary = ptx.execute_default_processing(export_dir=export_folder,
                                                                         plot_name=base_file_name, cell_size=cell_size,
                                                                         plot_radius=plot_radius,
                                                                         max_height=max_grid_height,
                                                                         max_occlusion=max_occlusion,
                                                                         sigma1=0, min_pad_foliage=.01,
                                                                         max_pad_foliage=6)
            profile['PLT_CN'] = base_file_name
            profile['CBD'] = canopy_model.predict(profile, lidar_value_col='PAD', height_col='HT', plot_id_col='PLT_CN')

            profile.to_csv(export_folder / 'PAD_Profile' / (base_file_name + '.csv'), index=False)

            print("Finished file ", i, " of ", len(files), " in ", round(time.time() - start_time, 3), " seconds")
            i += 1

        print("Finished pre-processing all files in ", round(time.time() - start_time_all), " seconds \n")

    print('Starting fuel and fire behavior summaries...')
    start_time = time.time()

    # Read CBD profiles from output csv files
    profile_paths = get_files_list(export_folder / 'PAD_Profile', '.csv', recursive=False)
    profiles = []
    for profile_path in profile_paths:
        profile = pd.read_csv(profile_path)
        profiles.append(profile)
    profiles = pd.concat(profiles)

    # Summarize height bins
    profiles['HEIGHT_BIN'] = pd.cut(profiles['HT'], bins=[.2,1,2,5,999], labels=['LOAD_02T1','LOAD_1T2','LOAD_2T5','LOAD_5T999'], include_lowest=True, right=False)
    bin_summary = profiles.pivot_table(index='PLT_CN',columns='HEIGHT_BIN',values='CBD',aggfunc='sum',observed=False) * cell_size
    bin_summary.columns = bin_summary.columns.values.astype(str)
    profiles = profiles[profiles['HT'] >= min_height]

    # Get fuel strata gap, effective CBD, and other summary values
    summary = voxelmon.utils.summarize_profiles(profiles, min_height=min_height)
    summary = summary.set_index('PLT_CN')

    # Add profile data
    summary = summary.join(bin_summary, how='left')

    # Add field data
    summary = summary.join(field_summary, how='left')

    # Add other lidar data
    summary_paths = get_files_list(export_folder / 'Plot_Summary', '.csv', recursive=False)
    lidar_summaries = pd.concat([pd.read_csv(path) for path in summary_paths])
    lidar_summaries = lidar_summaries.set_index('PLT_CN')
    summary = summary.join(lidar_summaries, how='inner')

    # Model fire with Behave (pyrothermel)
    summary['CHAR_SAVR'] = 0.
    summary['CHAR_LOAD_DEAD'] = 0.
    summary['CHAR_LOAD_LIVE'] = 0.
    summary['CHAR_LOAD_TOTAL'] = 0.
    behave_results = []
    for plotname in summary.index:
        fm = pyrothermel.FuelModel.from_existing(summary.loc[plotname, 'SURFACE_CLASS'])
        bd = fm.bulk_density

        load_coef = summary.loc[plotname, 'SURFACE_LOAD_COEF']
        if np.isfinite(load_coef):
            fm.fuel_load_one_hour *= load_coef
            fm.fuel_load_ten_hour *= load_coef
            fm.fuel_load_hundred_hour *= load_coef
            fm.fuel_load_live_herbaceous *= load_coef
            fm.fuel_load_live_woody *= load_coef
        bd_coef = summary.loc[plotname, 'SURFACE_BD_COEF']
        if np.isfinite(bd_coef):
            fm.bulk_density = bd_coef * bd
        else:
            fm.bulk_density = bd
        load_dead, load_live = fm.characteristic_load()
        savr = fm.characteristic_savr()
        summary.loc[plotname, 'CHAR_SAVR'] = savr
        summary.loc[plotname, 'CHAR_LOAD_DEAD'] = load_dead
        summary.loc[plotname, 'CHAR_LOAD_LIVE'] = load_live
        summary.loc[plotname, 'CHAR_LOAD_TOTAL'] = load_dead + load_live
        ms = pyrothermel.MoistureScenario.from_existing(1, 2)
        up = pyrothermel.UnitsPreset.metric()
        run = pyrothermel.PyrothermelRun(fm, ms, wind_speed, units_preset=up, wind_input_mode='twenty_foot',
                                         canopy_base_height=summary.loc[plotname, 'FSG'],
                                         canopy_bulk_density=summary.loc[plotname, 'CBD'],
                                         canopy_cover=summary.loc[plotname, 'CANOPY_COVER'],
                                         canopy_height=summary.loc[plotname, 'CH'],
                                         canopy_ratio=summary.loc[plotname, 'CR'],
                                         slope=summary.loc[plotname, 'TERRAIN_SLOPE'],
                                         aspect=summary.loc[plotname, 'TERRAIN_ASPECT'])
        run.run_surface_fire_in_direction_of_max_spread()
        result = run.run_crown_fire_scott_and_reinhardt()
        result['TORCHING_INDEX'] = run.calculate_torching_index(max_wind_speed=1000)
        result['CROWNING_INDEX'] = run.calculate_crowning_index(max_wind_speed=1000)
        result['PLT_CN'] = plotname
        behave_results.append(result)
    behave_results = pd.DataFrame(behave_results)
    behave_results.columns = [col.upper() for col in behave_results.columns]
    behave_results = behave_results.set_index('PLT_CN')

    summary = summary.join(behave_results, how='inner')

    summary.to_csv(export_folder.joinpath('results_summary.csv'))

    print("Finished fuel and fire behavior summaries in ", round(time.time() - start_time), " seconds \n")
    print("Starting figure outputs...")
    start_time = time.time()

    # Generate output figures
    if generate_figures:
        import polars as pl
        for plotname in profiles['PLT_CN'].unique():
            # profile_path = [Path(filename) for filename in tls_summary_files if plotname in filename][0]
            dempath = Path('/'.join([str(export_folder), 'DEM', plotname + '.csv']))
            pointspath = Path('/'.join([str(export_folder), 'Points', plotname + '.csv']))
            profile = profiles[profiles['PLT_CN'] == plotname]
            pts = pl.read_csv(pointspath)
            demPts = pl.read_csv(dempath)
            f, [ax1, ax2] = plt.subplots(ncols=2, sharey=True, figsize=[8, 4])
            [arr, arr_extents] = plot_side_view(pts, direction=3, demPtsNormalize=demPts, returnData=True)
            ax1.imshow(arr, extent=arr_extents, aspect=2)
            ax2.axhspan(summary.loc[plotname, 'FSG_H1'], summary.loc[plotname, 'FSG_H2'], color='yellow', alpha=0.3,
                        label='Fuel Strata Gap (FSG)')
            ax2.plot(profile['CBD'], profile['HT'], label='Canopy Bulk Density (kg/m^3)')
            ax2.axvline(summary.loc[plotname, 'CBD'], linestyle='--', color='black',
                        label='Effective Canopy Bulk Density')
            ax2.axvline(.011, linestyle='--', color='yellow', label='FSG Cutoff')
            ymax = min(max(ax1.get_ylim()[1], ax2.get_ylim()[1], 14),max_grid_height)
            ax1.set_ylim([0, ymax])
            ax2.set_ylim([0, ymax])
            if cbd_axis_limit is not None:
                ax2.set_xlim([0, cbd_axis_limit])
            ax2.set_yticks(ax1.get_yticks())
            ax1.text(0, 1.1, plotname, transform=ax1.transAxes, fontsize=12, ha='left')
            ax1.set_ylabel('Height (m)')
            ax1.set_xlabel('Easting (m)')
            ax2.set_xlabel('Canopy Bulk Density (kg/m^3)')
            ax2.legend(loc="upper right", prop={'size': 'small'})
            table_data = [['Effective CBD', 'Fuel Strata Gap', 'Fuel 0.2m-1m',
                           'Fuel 1m-2m', 'Fuel 2m-5m', 'Fuel >5m'],
                          [summary.loc[plotname, 'CBD'].round(4),
                           summary.loc[plotname, 'FSG'].round(1),
                           summary.loc[plotname, 'LOAD_02T1'].round(4),
                           summary.loc[plotname, 'LOAD_1T2'].round(4),
                           summary.loc[plotname, 'LOAD_2T5'].round(4),
                           summary.loc[plotname, 'LOAD_5T999'].round(4)],
                          ['kg/m^3', 'm', 'kg/m^2', 'kg/m^2', 'kg/m^2', 'kg/m^2']]
            table_data = np.array(table_data).T
            ax2.table(cellText=table_data, colLabels=['Name', 'Value', 'Units'], cellLoc='center',
                      bbox=[1.1, 0, .75, 1], colWidths=[.5, .25, .25])
            # ax2.text(1.1, -.1, f"Potential fire behavior based on \n{wind_speed}km/hr wind; 'very low' moisture",
            #          transform=ax2.transAxes, fontsize=8, ha='left')
            f.tight_layout(pad=2)
            plt.savefig(export_folder.joinpath(plotname + '.png'), dpi=300)
            # plt.show()
            print("Finished figure for ", plotname)

    print("Finished figure outputs in ", round(time.time() - start_time), " seconds \n")
    print("Finished all processing in ", round(time.time() - start_time_all), " seconds")
    print("Done. You can close this window.")



def main():
    parser = argparse.ArgumentParser(description="Process canopy data and generate summaries.")

    parser.add_argument("input_folder", type=str,
                        help="Path to the input data folder")
    parser.add_argument("export_folder", type=str,
                        help="Path to export data folder")
    parser.add_argument("field_summary_path", type=str,
                        help="Path file containing field summary data, see ./template_files/surface_fuel_summary.csv")
    parser.add_argument("canopy_model_path", type=str,
                        help="Path to the canopy LMA model file, see ./template_files/canopy_lma_model.xlsx")

    parser.add_argument("--cell_size", type=float, default=0.1, help="Size of the grid cells")
    parser.add_argument("--plot_radius", type=float, default=11.3, help="Radius for plotting")
    parser.add_argument("--min_height", type=float, default=0.2, help="Minimum height threshold")
    parser.add_argument("--max_grid_height", type=float, default=30.0, help="Maximum height of the grid")
    parser.add_argument("--max_occlusion", type=float, default=0.8, help="Maximum occlusion threshold, 0.-1.")
    parser.add_argument("--cbd_axis_limit", type=str, default='', help="X axis limit for CBD plot, leave blank for no limit")
    parser.add_argument("--wind_speed", type=float, default=30.0, help="Wind speed (km/hr) used in fire behavior calculation")
    parser.add_argument("--wind_direction", type=float, default=0.0, help="Wind direction used in fire behavior calculation")

    parser.add_argument("--no-process", action="store_false", dest="do_processing", help="Disable the processing step")
    parser.add_argument("--no-generate_figures", action="store_false", dest="generate_figures", help="Disable figure generation")
    parser.set_defaults(do_processing=True, generate_figures=True)

    args = parser.parse_args()

    # Call the function using the parsed arguments
    process(
        input_folder=args.input_folder,
        export_folder=args.export_folder,
        field_summary_path=args.field_summary_path,
        canopy_model_path=args.canopy_model_path,
        cell_size=args.cell_size,
        plot_radius=args.plot_radius,
        min_height=args.min_height,
        max_grid_height=args.max_grid_height,
        max_occlusion=args.max_occlusion,
        cbd_axis_limit=args.cbd_axis_limit,
        wind_speed=args.wind_speed,
        wind_direction=args.wind_direction,
        do_processing=args.do_processing,
        generate_figures=args.generate_figures
    )

if __name__ == "__main__":
    main()
