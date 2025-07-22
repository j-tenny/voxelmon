This folder contains templates to facilate the process_predict_ptx scripts

## surface_fuel_summary

This file describes surface fuel and metadata for each scan. The only required columns are PLT_CN and SURFACE_CLASS.

- PLT_CN: unique key that matches the name of a ptx file (no extension)
- LATITUDE: geographic latitude
- LONGITUDE: geographic longitude
- SURFACE_CLASS: identifier for a surface fuel model from the Scott and Burgan 40 standard surface fuel models.
- SURFACE_BD_COEF: A multiplier applied to surface fuelbed bulk density. Defaults to 1. Values between 0.5 and 1 will typically result in higher rate of spread and flame length. Values greater than 1 will typically result in lower rate of spread and flame length. Values less than 0.5 or greater than 2.0 are not recommended.
- SURFACE_LOAD_COEF: A multiplier applied to surface fuel load. Defaults to 1. Values between 0 and 1 will result in lower rate of spread and flame length. Values greater than 1 will result in higher rate of spread and flame length.


## canopy_lma_model

This file describes species composition along the vertical profile and provides leaf-mass-per-area values for each species.
The number of rows and columns is flexible to accomodate any number of species and height intervals. The first row 
contains model parameters: a model intercept term (usually 0), followed by the calibrated leaf (fuel) mass per area for
each species in the table. The second row contains headers. These are only for readability and are not used in the program.
In the remaining rows, the first column must contain height in meters. The remaining columns describe the proportion of
foliage contributed by each species at a given height. CHAP.csv, PJO.csv, and PPO.csv represent calibrated models
for chaparral, pinyon-juniper-oak, and ponderosa-pine-oak for a study area in the Central Arizona Mogollon Highlands,
as described by Tenny et al 2025.