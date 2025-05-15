## voxelmon
This package produces voxelized leaf area density metrics from lidar which can be used to produce vegetation metrics for wildfire risk modeling, 
treatment effect monitoring, hydrology analysis, wildlife habitat suitability, and more. Currently the package is compatible with single- and multiple- scan data from the Leica BLK 360 as well as ALS and UAV lidar.

## Installation
`pip install git+https://github.com/j-tenny/voxelmon.git`

## Usage
This code will process a single TLS scan in .ptx format and create files within the export folder including a 3D grid 
with leaf area density estimates, a vertical profile of leaf area density, and plot summary metrics. The 3D grid is 
represented with a row for each grid cell with columns for x, y, z coordinates, leaf (plant) area density, occlusion, etc.

For more detailed examples, see the example_scripts folder.

```
import voxelmon
ptx_file = 'Data/scan1.ptx'
export_dir = 'Data/'
ptx = voxelmon.TLS_PTX(ptxFile, apply_translation=True, apply_rotation=True, drop_null=False)
grid, profile, plot_summary = ptx.execute_default_processing(export_dir=export_dir, plot_name='scan1', cell_size=0.1,
                                                             plot_radius=11.3, max_height=50, max_occlusion=0.75,
                                                             sigma1=0, min_pad_foliage=.01, max_pad_foliage=6)
```

Scripts to recreate the analysis from the publication "Canopy and surface fuels measurement using 
terrestrial lidar single-scan approach in the Mogollon highlands of Arizona" are located in the branch final_analysis_tenny_et_al_2025. The
master branch will be updated with newer algorithms and pipelines for additional data sources. 

## Contact

Contact jt893@nau.edu with any questions. 