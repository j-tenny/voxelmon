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

        process(*args)


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
    field_summary = pd.read_csv(field_summary_path, index_col='EVENT_ID')
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
            print("Starting file ", i, " of ", len(files))
            base_file_name = os.path.splitext(os.path.basename(ptx_file))[0].split('-')[0]

            ptx = TLS_PTX(ptx_file, apply_translation=False, apply_rotation=True, drop_null=False)
            grid, profile, plot_summary = ptx.execute_default_processing(export_dir=export_folder,
                                                                         plot_name=base_file_name, cell_size=cell_size,
                                                                         plot_radius=plot_radius,
                                                                         max_height=max_grid_height,
                                                                         max_occlusion=max_occlusion,
                                                                         sigma1=0, min_pad_foliage=.01,
                                                                         max_pad_foliage=6)
            profile['EVENT_ID'] = base_file_name
            profile['CANOPY_CLASS'] = field_summary.loc[base_file_name, 'CANOPY_CLASS']
            profile['CBD'] = canopy_model.predict(profile, lidar_value_col='pad', height_col='height',
                                                  plot_id_col='EVENT_ID')

            profile.to_csv(export_folder / 'PAD_Summary' / (base_file_name + '.csv'), index=False)

            print("Finished file ", i, " of ", len(files), " in ", round(time.time() - start_time, 3), " seconds")
            i += 1

        print("Finished all files in ", round(time.time() - start_time_all), " seconds")

    # Read CBD profiles from output csv files
    profile_paths = get_files_list(export_folder / 'PAD_Summary', '.csv', recursive=False)
    profiles = []
    for profile_path in profile_paths:
        profile = pd.read_csv(profile_path)
        profiles.append(profile)
    profiles = pd.concat(profiles)
    profiles = profiles[profiles['height'] >= min_height]

    # Get fuel strata gap, effective CBD, and other summary values
    summary = voxelmon.utils.summarize_profiles(profiles, bin_height=cell_size, min_height=min_height,
                                                fsg_threshold=.011, cbd_col='CBD', height_col='height', pad_col='pad',
                                                occlusion_col='occluded', plot_id_col='EVENT_ID')

    # Join surface fuel data
    summary = summary.set_index('EVENT_ID')
    summary = summary.join(field_summary, how='inner')
    summary['LOAD_TOTAL'] = summary['LOAD_FDF'] + summary['LOAD_10HR'] + summary['LOAD_100HR'] + summary['LOAD_LH'] + \
                            summary['LOAD_LW']

    # Model fire with Behave (pyrothermel)
    behave_results = []
    for plotname in summary.index:
        fm = pyrothermel.FuelModel.from_existing(summary.loc[plotname, 'SURFACE_CLASS'])
        bd = fm.bulk_density
        load_1hr = summary.loc[plotname, 'LOAD_FDF']
        if np.isfinite(load_1hr):
            fm.fuel_load_one_hour = load_1hr
        load_10hr = summary.loc[plotname, 'LOAD_10HR']
        if np.isfinite(load_10hr):
            fm.fuel_load_ten_hour = load_10hr
        load_100hr = summary.loc[plotname, 'LOAD_100HR']
        if np.isfinite(load_100hr):
            fm.fuel_load_hundred_hour = load_100hr
        load_lh = summary.loc[plotname, 'LOAD_LH']
        if np.isfinite(load_lh):
            fm.fuel_load_live_herbaceous = load_lh
        load_lw = summary.loc[plotname, 'LOAD_LW']
        if np.isfinite(load_lw):
            fm.fuel_load_live_woody = load_lw
        depth = summary.loc[plotname, 'FUELBED_HT'] / 100
        if np.isfinite(depth):
            fm.fuel_bed_depth = depth
        else:
            fm.bulk_density = bd

        ms = pyrothermel.MoistureScenario.from_existing(1, 2)
        up = pyrothermel.UnitsPreset.metric()
        run = pyrothermel.PyrothermelRun(fm, ms, wind_speed, wind_direction=wind_direction,
                                         units_preset=up, wind_input_mode='twenty_foot',
                                         canopy_base_height=summary.loc[plotname, 'fsg'],
                                         canopy_bulk_density=summary.loc[plotname, 'cbd_max'],
                                         canopy_cover=summary.loc[plotname, 'canopy_cover'],
                                         canopy_height=summary.loc[plotname, 'canopy_height'],
                                         canopy_ratio=summary.loc[plotname, 'canopy_ratio'],
                                         slope=summary.loc[plotname, 'terrain_slope'],
                                         aspect=summary.loc[plotname, 'terrain_aspect'])
        run.run_surface_fire_in_direction_of_max_spread()
        result = run.run_crown_fire_scott_and_reinhardt()
        result['torching_index'] = run.calculate_torching_index(max_wind_speed=1000)
        result['crowning_index'] = run.calculate_crowning_index(max_wind_speed=1000)
        result['EVENT_ID'] = plotname
        behave_results.append(result)
    behave_results = pd.DataFrame(behave_results)
    behave_results = behave_results.set_index('EVENT_ID')

    summary = summary.join(behave_results, how='inner')

    summary.to_csv(export_folder / 'results_summary.csv')

    # Generate output figures
    if generate_figures:
        import polars as pl
        for plotname in profiles['EVENT_ID'].unique():
            # profile_path = [Path(filename) for filename in tls_summary_files if plotname in filename][0]
            dempath = Path('/'.join([str(export_folder), 'DEM', plotname + '.csv']))
            pointspath = Path('/'.join([str(export_folder), 'Points', plotname + '.csv']))
            profile = profiles[profiles['EVENT_ID'] == plotname]
            pts = pl.read_csv(pointspath)
            demPts = pl.read_csv(dempath)
            f, [ax1, ax2] = plt.subplots(ncols=2, sharey=True, figsize=[8, 4])
            [arr, arr_extents] = plot_side_view(pts, direction=3, demPtsNormalize=demPts, returnData=True)
            ax1.imshow(arr, extent=arr_extents, aspect=2)
            ax2.axhspan(summary.loc[plotname, 'fsg_h1'], summary.loc[plotname, 'fsg_h2'], color='yellow', alpha=0.3,
                        label='Fuel Strata Gap (FSG)')
            ax2.plot(profile['CBD'], profile['height'], label='Canopy Bulk Density (kg/m^3)')
            ax2.axvline(summary.loc[plotname, 'cbd_max'], linestyle='--', color='black',
                        label='Max Canopy Bulk Density')
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
            table_data = [
                ['Max CBD', 'Fuel Strata Gap', 'Surface Fuel Load', 'Fuelbed Depth', 'Spread Rate', 'Intensity',
                 'Flame Length', 'Torching Index', 'Crowning Index'],
                [summary.loc[plotname, 'cbd_max'].round(4), summary.loc[plotname, 'fsg'].round(1),
                 summary.loc[plotname, 'LOAD_TOTAL'].round(2), summary.loc[plotname, 'FUELBED_HT'].round(1),
                 summary.loc[plotname, 'spread_rate'].round(3), summary.loc[plotname, 'fireline_intensity'].round(1),
                 summary.loc[plotname, 'flame_length'].round(1), summary.loc[plotname, 'torching_index'],
                 summary.loc[plotname, 'crowning_index']],
                ['kg/m^3', 'm', 'kg/m^2', 'cm', 'km/hr', 'kW/m', 'm', 'km/hr', 'km/hr']]
            table_data = np.array(table_data).T
            ax2.table(cellText=table_data, colLabels=['Name', 'Value', 'Units'], cellLoc='center',
                      bbox=[1.1, 0, .75, 1], colWidths=[.5, .25, .25])
            ax2.text(1.1, -.1, f"Potential fire behavior based on \n{wind_speed}km/hr wind; 'very low' moisture",
                     transform=ax2.transAxes, fontsize=8, ha='left')
            f.tight_layout(pad=2)
            plt.savefig(export_folder.joinpath(plotname + '.png'), dpi=300)
            plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    stacked = QStackedWidget()
    form = ParameterForm(stacked)
    stacked.addWidget(form)
    stacked.setWindowTitle("Voxelmon Process PTX Predict Fire Behavior")
    stacked.resize(600, 500)
    stacked.show()
    sys.exit(app.exec_())

