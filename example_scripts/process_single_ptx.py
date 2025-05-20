# Process plant area density for a single ptx file
import voxelmon
import time
time_s = time.time()
ptx = voxelmon.TLS_PTX('../test_data/T1423071203- Med Density 1.ptx')
ptx.execute_default_processing('../test_outputs/','T1423071203')
print(time.time()-time_s)