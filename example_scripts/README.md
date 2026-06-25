## build_canopy_profile_custom.R
Produce canopy bulk density profiles from conventional tree measurements with custom allometric equations. Provides input 
for calibrating leaf-mass-per-area values. This script is written for R software. 

## process_multi_ptx.py
Simple example to process plant area density for a group of ptx files representing one plot in multiple scan mode. 
Scans must be registered/aligned.

## process_multi_ptx_dir.py
Process plant area density for all ptx files in a directory where there are multiple scans per plot.

## process_predict_ptx_dir.py
Command line application to process plant area density, then predict canopy bulk density and potential fire behavior by applying existing CBD model.
Each ptx file is treated as an independent plot.
Fire behavior estimates require pyrothermel `pip install pyrothermel`

## process_predict_ptx_dir.py
Graphical user interface to process plant area density, then predict canopy bulk density and potential fire behavior by applying existing CBD model.
Each ptx file is treated as an independent plot.
Fire behavior estimates require pyrothermel `pip install pyrothermel`

## process_ptx_gui.py
Graphical user interface to process plant area density for all ptx files in a directory. Each ptx file is treated as an independent plot.

## process_single_ptx.py
Simple example to process plant area density for a single ptx file

## process_single_ptx_dir.py
Process plant area density for all ptx files in a directory. Each ptx file is treated as an independent plot.

## train_test_multiple_scan_ptx.py
Train and test a canopy bulk density profile model for multiple-scan lidar. Uses field data to produce calibrated 
leaf-mass-per-area estimates as in Tenny et al 2025. File and column identifiers will need significant updates.
Requires pymc and scikit-learn libraries `conda install pymc scikit-learn`

## train_test_coeficients.py
Train and test a canopy bulk density profile model for single-scan lidar. Uses field data to produce calibrated 
leaf-mass-per-area estimates as in Tenny et al 2025. File and column identifiers will need significant updates.

## viewer.py
View processed plant area density grid in 3D
Requires pyvista `pip install pyvista`
