import voxelmon
als = voxelmon.ALS('../test_data/USGS_LPC_CA_SierraNevada_B22_10SFJ6362.laz')
flightpath = als.estimate_flightpath()
flightpath.write_csv('../test_outputs/flightpath.csv')
als.execute_default_processing('../test_outputs/','ALS_test',cell_size=3)