# Process plant area density and predict canopy bulk density and potential fire behavior by applying existing model
# Fire behavior estimates require pyrothermel `pip install pyrothermel`
import pandas as pd
import time
import numpy as np
from pathlib import Path
import polars
import pyvista
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

        add_file_field("grid_path", is_folder=False)
        add_spin_field("min_value", .05)
        add_spin_field("max_value", 6)
        add_spin_field("min_hag", .1)

        submit_btn = QPushButton("Submit")
        submit_btn.clicked.connect(self.on_submit)
        layout.addRow(submit_btn)

        self.setLayout(layout)

    def on_submit(self):
        import polars as pl
        # Gather all inputs as positional args
        ui_input = dict([(k,v()) for k, v in self.fields.items()])
        kwargs = {}

        grid_path = Path(ui_input['grid_path'])
        kwargs['grid'] = pl.read_csv(grid_path)
        basename = grid_path.name
        dem_path = grid_path.parents[1] / 'DEM' / basename
        kwargs['dem'] = pl.read_csv(dem_path)

        points_path = grid_path.parents[1] / 'Points' / basename
        if points_path.exists():
            kwargs['points'] = pl.read_csv(points_path)
        else:
            kwargs['points'] = None

        kwargs['min_value_leaf'] = ui_input['min_value']
        kwargs['max_value_leaf'] = ui_input['max_value']
        kwargs['min_hag'] = ui_input['min_hag']

        kwargs['clip_extents'] = 'auto'
        kwargs['value_name'] = 'PAD'


        self.hide()

        #voxelmon.visualize_voxels(**kwargs)

        QTimer.singleShot(0, lambda: voxelmon.visualize_voxels(**kwargs))


class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.form = ParameterForm(self)
        self.addWidget(self.form)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    stacked = QStackedWidget()
    form = ParameterForm(stacked)
    stacked.addWidget(form)
    stacked.setWindowTitle("Voxelmon Viewer")
    stacked.resize(600, 500)
    stacked.show()
    sys.exit(app.exec_())

