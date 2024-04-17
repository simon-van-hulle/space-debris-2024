import numpy as np

area_arr = np.loadtxt('area_final.dat')
area_std_dev = np.std(area_arr)
area_mean = np.mean(area_arr)
print('Area standard deviation:', area_std_dev)
print('Area mean:', area_mean)