## build_canopy_profile_custom.R
Produce canopy bulk density profiles from conventional tree measurements with custom allometric equations. Provides input 
for calibrating leaf-mass-per-area values. This script is written for R software. 

## build_canopy_profile_nsvb
Produce canopy bulk density profiles from conventional tree measurements with allometric equations from NSVB (National 
Scale Volume Biomass estimators). Provides input for calibrating leaf-mass-per-area values.
Requries nsvb library `pip install nsvb`

## process_als.py
Use path-tracing approach to process 3D plant area density for an ALS data tile. Note, estimate_flightpath() has 
resulted in mixed quality results. Carefully check flightpath for realism. Consider using ALS.simple_pad() as an 
alternative to ALS.execute_default_processing() especially if pts/m^2 < 30. simple_pad() assumes pulses are directed
straight down and is best suited for large voxels (e.g. 10m x 10m x 2m).

## process_predict_ptx_dir.py
Process plant area density and predict canopy bulk density and potential fire behavior by applying existing CBD model
Fire behavior estimates require pyrothermel `pip install pyrothermel`

## process_single_ptx.py
Simple example to process plant area density for a single ptx file

## process_single_ptx_dir.py
Process plant area density for all ptx files in a directory. It is assumed each file is an independent plot location.

## train_test_multiple_scan_ptx.py
Train and test a canopy bulk density profile model for multiple-scan lidar. Uses field data to produce calibrated 
leaf-mass-per-area estimates as in Tenny et al 2025. File and column identifiers will need significant updates.
Requires pymc and scikit-learn libraries `conda install pymc scikit-learn`

## train_test_single_scan_ptx.py
Train and test a canopy bulk density profile model for single-scan lidar. Uses field data to produce calibrated 
leaf-mass-per-area estimates as in Tenny et al 2025. File and column identifiers will need significant updates.

## viewer.py
View processed plant area density grid in 3D
Requires pyvista `pip install pyvista`
