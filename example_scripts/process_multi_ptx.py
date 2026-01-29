# Process plant area density for a single plot consisting of multiple scans
import voxelmon
import time
time_s = time.time()
ptx = voxelmon.TLS_PTX_Group(['../test_data/T1423071203- Med Density 1.ptx',
                              '../test_data/T1423071203- Med Density 2.ptx',
                              '../test_data/T1423071203- Med Density 3.ptx',
                              '../test_data/T1423071203- Med Density 4.ptx'])
grid, profile, summary = ptx.execute_default_processing('../test_outputs/','T1423071203')
print(time.time()-time_s)