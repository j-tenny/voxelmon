This folder contains templates to facilate the process_predict_ptx scripts

## surface_fuel_summary

This file describes surface fuel and metadata for each scan. The only required columns are EVENT_ID and SURFACE_CLASS.

- EVENT_ID: unique key that matches the name of a ptx file (no extension)
- LATITUDE: geographic latitude
- LONGITUDE: geographic longitude
- CANOPY_CLASS: label applied to profile data. Could be used to specify a different LMA model for each scan.
- SURFACE_CLASS: identifier for a surface fuel model from the Scott and Burgan 40 standard surface fuel models.
- LOAD_LH: live herbaceous fuel load, kg/m^2
- LOAD_LW: live woody fuel load, kg/m^2
- LOAD_FDF: fine dead fuel load (1-hr fuel + available litter), kg/m^2
- LOAD_10HR: 10-hr dead fuel load, kg/m^2
- LOAD_100HR: 100-hr dead fuel load, kg/m^2
- FUELBED_HT: Rothermel surface fuel depth, cm

## canopy_lma_model

This file describes species composition along the vertical profile and provides leaf-mass-per-area values for each species.
The number of rows and columns is flexible to accomodate any number of species and height intervals. The first row 
contains model parameters: a model intercept term (usually 0), followed by the calibrated leaf (fuel) mass per area for
each species in the table. The second row contains headers. These are only for readability and are not used in the program.
In the remaining rows, the first column must contain height in meters. The remaining columns describe the proportion of
foliage contributed by each species at a given height. CHAP.csv, PJO.csv, and PPO.csv represent calibrated models
for chaparral, pinyon-juniper-oak, and ponderosa-pine-oak for a study area in the Central Arizona Mogollon Highlands,
as described by Tenny et al 2025.