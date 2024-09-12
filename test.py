import scipy.io as sio
import numpy as np
from scipy.io import loadmat
import pandas as pd

# 文件路径
file_pathnpy = '/home/alk/Data/testMDD/507_Depression_REST-epo.npy'
file_pathnpy2 = '/home/alk/Data/testMDD/507_Depression_REST-epo-times.npy'

# 12个event, event      4 个epoch
npy_data = np.load(file_pathnpy, allow_pickle=True)
event1 = npy_data[0]
event2 = npy_data[1]
event3 = npy_data[2]
event4 = npy_data[3]
event5 = npy_data[4]
event6 = npy_data[5]
event7 = npy_data[6]
event8 = npy_data[7]
event9 = npy_data[8]
event10 = npy_data[9]
event11 = npy_data[10]
event12 = npy_data[11]



npy_data2 = np.load(file_pathnpy2, allow_pickle=True)

print('1')


