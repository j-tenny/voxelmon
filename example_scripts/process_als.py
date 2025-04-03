import voxelmon
import time
time_start = time.time()
als = voxelmon.ALS('../test_data/USGS_LPC_CA_SierraNevada_B22_10SFJ6362.laz')
flightpath = als.estimate_flightpath()
flightpath.write_csv('../test_outputs/flightpath.csv')
grid, profile, summary = als.execute_default_processing('../test_outputs/','ALS_test',cell_size=2,sigma1=1,min_pad_foliage=0.1, max_pad_foliage=1)
print(time.time() - time_start)