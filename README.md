## voxelmon
This package produces voxelized leaf area density metrics from lidar which can be used to produce vegetation metrics for wildfire risk modeling, 
treatment effect monitoring, hydrology analysis, wildlife habitat suitability, and more. Currently the package is compatible with single- and multiple- scan data from the Leica BLK 360 as well as ALS and UAV lidar.

## Installation
If you are unfamiliar with Python, follow this [user guide](https://docs.google.com/document/d/1OuOK_Xj9kzQYp_oARr-3C10PqYRyHlgRpw89BaMAPE4/edit?usp=sharing)

If you are familiar with Python:
`cd [install directory]`
`git clone https://github.com/j-tenny/voxelmon.git`
`conda create -n [env_name]`
`pip install [install directory]`
`pip install requirements.txt`

Or `pip install git+https://github.com/j-tenny/voxelmon.git`

## Usage
If you are unfamiliar with Python, follow this [user guide](https://docs.google.com/document/d/1OuOK_Xj9kzQYp_oARr-3C10PqYRyHlgRpw89BaMAPE4/edit?usp=sharing)

If you are familiar with Python, look through the example_scripts folder (the folder contains its own readme).
As a simple example, this code will process a single TLS scan in .ptx format and create files within the export folder including a 3D grid 
with leaf area density estimates, a vertical profile of leaf area density, and plot summary metrics. The export 3D grid is
formatted as .csv file with a row for each grid cell and columns for x, y, z coordinates, leaf (plant) area density, occlusion, etc.
This allows the file to be opened in CloudCompare.

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
