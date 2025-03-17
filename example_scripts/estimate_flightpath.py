import voxelmon
import time
als = voxelmon.ALS('C:\\Users\\john1\\Documents\\Github\\dtm_utils\\test_data\\als_creek.laz')

start_time = time.time()
flight_points = als.estimate_flightpath(fit_line=True)
print(time.time()-start_time)
flight_points.write_csv('flight_points.csv',)