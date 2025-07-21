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
import io
import os
import contextlib
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QDoubleSpinBox, QCheckBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QFormLayout, QSpinBox, QTextEdit, QStackedWidget
)
from PyQt5.QtCore import QTimer

class ParameterForm(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()
        self.fields = {}

        def add_file_field(name, is_folder=False):
            line_edit = QLineEdit()
            browse = QPushButton("Browse")
            def browse_file():
                path = QFileDialog.getExistingDirectory(self, "Select Folder") if is_folder else QFileDialog.getOpenFileName(self, "Select File")[0]
                if path: line_edit.setText(path)
            browse.clicked.connect(browse_file)
            hlayout = QHBoxLayout()
            hlayout.addWidget(line_edit)
            hlayout.addWidget(browse)
            container = QWidget(); container.setLayout(hlayout)
            self.fields[name] = lambda: line_edit.text()
            layout.addRow(name.replace("_", " ").title(), container)

        def add_spin_field(name, default):
            spin = QDoubleSpinBox(); spin.setValue(default)
            self.fields[name] = spin.value
            layout.addRow(name.replace("_", " ").title(), spin)

        def add_bool_field(name, default):
            checkbox = QCheckBox(); checkbox.setChecked(default)
            self.fields[name] = checkbox.isChecked
            layout.addRow(name.replace("_", " ").title(), checkbox)

        add_file_field("input_folder", is_folder=True)
        add_file_field("export_folder", is_folder=True)
        add_file_field("field_summary_path")
        add_file_field("canopy_model_path")
        add_spin_field("cell_size", 0.1)
        add_spin_field("plot_radius", 11.3)
        add_spin_field("min_height", 0.2)
        add_spin_field("max_grid_height", 30)
        add_spin_field("max_occlusion", 0.75)

        cbd_input = QLineEdit(); cbd_input.setPlaceholderText("Leave blank for None")
        self.fields["cbd_axis_limit"] = lambda: float(cbd_input.text()) if cbd_input.text() else None
        layout.addRow("CBD Axis Limit", cbd_input)

        add_spin_field("wind_speed", 30.0)
        add_spin_field("wind_direction", 270.0)
        add_bool_field("process", True)
        add_bool_field("generate_figures", True)

        submit_btn = QPushButton("Submit")
        submit_btn.clicked.connect(self.on_submit)
        layout.addRow(submit_btn)

        self.setLayout(layout)

    def on_submit(self):
        # Gather all inputs as positional args
        args = [v() for k, v in self.fields.items()]

        # Show console page with output
        # console_page = ConsolePage()
        # self.stacked_widget.addWidget(console_page)
        # self.stacked_widget.setCurrentWidget(console_page)

        self.close()

        QTimer.singleShot(0, lambda: process(*args))


class ConsolePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        output = QTextEdit()
        output.setReadOnly(True)
        layout.addWidget(output)
        self.setLayout(layout)

class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.form = ParameterForm(self)
        self.addWidget(self.form)

def process(input_folder, export_folder, field_summary_path, canopy_model_path,
            cell_size, plot_radius,min_height, max_grid_height, max_occlusion, cbd_axis_limit,
            wind_speed, wind_direction, process,generate_figures):
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

    if process:
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
            profile['CANOPY_CLASS'] = field_summary.loc[base_file_name, 'CANOPY_CLASS']
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
    profiles = profiles[profiles['HT'] >= min_height]

    # Get fuel strata gap, effective CBD, and other summary values
    summary = voxelmon.utils.summarize_profiles(profiles, min_height=min_height)
    summary = summary.set_index('PLT_CN')

    # Add field data
    summary = summary.join(field_summary, how='inner')

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
            ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1], 14)
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
            table_data = [['Effective CBD', 'Fuel Strata Gap', 'Effective Surface Load', 'Spread Rate', 'Intensity',
                           'Flame Length', 'Torching Index', 'Crowning Index'],
                          [summary.loc[plotname, 'CBD'].round(4), summary.loc[plotname, 'FSG'].round(1),
                           summary.loc[plotname, 'CHAR_LOAD_TOTAL'].round(2),
                           summary.loc[plotname, 'SPREAD_RATE'].round(3),
                           summary.loc[plotname, 'FIRELINE_INTENSITY'].round(1),
                           summary.loc[plotname, 'FLAME_LENGTH'].round(1), summary.loc[plotname, 'TORCHING_INDEX'],
                           summary.loc[plotname, 'CROWNING_INDEX']],
                          ['kg/m^3', 'm', 'kg/m^2', 'km/hr', 'kW/m', 'm', 'km/hr', 'km/hr']]
            table_data = np.array(table_data).T
            ax2.table(cellText=table_data, colLabels=['Name', 'Value', 'Units'], cellLoc='center',
                      bbox=[1.1, 0, .75, 1], colWidths=[.5, .25, .25])
            ax2.text(1.1, -.1, f"Potential fire behavior based on \n{wind_speed}km/hr wind; 'very low' moisture",
                     transform=ax2.transAxes, fontsize=8, ha='left')
            f.tight_layout(pad=2)
            plt.savefig(export_folder.joinpath(plotname + '.png'), dpi=300)
            plt.show()
            print("Finished figure for ", plotname)

    print("Finished figure outputs in ", round(time.time() - start_time), " seconds \n")
    print("Finished all processing in ", round(time.time() - start_time_all), " seconds")
    print("Done. You can close this window.")

if __name__ == "__main__":
    print("Process PTX files and predict fire behavior \n**Leave this window open!**")
    app = QApplication(sys.argv)
    stacked = QStackedWidget()
    form = ParameterForm(stacked)
    stacked.addWidget(form)
    stacked.setWindowTitle("Voxelmon Process PTX Predict Fire Behavior")
    stacked.resize(600, 500)
    stacked.show()
    #sys.exit(app.exec_())

