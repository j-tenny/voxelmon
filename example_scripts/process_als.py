# Use path-tracing approach to process 3D plant area density for an ALS data tile
# Note, estimate_flightpath() has resulted in mixed quality results. Carefully check flightpath for realism.
# Consider using ALS.simple_pad() as an alternative to ALS.execute_default_processing() especially if pts/m^2 < 30
import voxelmon
import time
time_start = time.time()
als = voxelmon.ALS('../test_data/USGS_LPC_CA_SierraNevada_B22_10SFJ6362.laz')
flightpath = als.estimate_flightpath()
flightpath.write_csv('../test_outputs/flightpath.csv')
grid, profile, summary = als.execute_default_processing('../test_outputs/','ALS_test',cell_size=2,sigma1=1,min_pad_foliage=0.1, max_pad_foliage=1)
#xr_grid = simple_pad(bin_size_xy = 10, bin_size_z = 2, min_height = 1, extinction_coefficient=.5, return_type='xarray')
print(time.time() - time_start)