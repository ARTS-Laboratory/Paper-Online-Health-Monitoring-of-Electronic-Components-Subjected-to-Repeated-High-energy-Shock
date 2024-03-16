import matplotlib.pyplot as plt
import numpy as np
import json
import os
import scipy
from numpy import ndarray
from scipy.signal import periodogram
"""
find half-sine time and maximum acceleration
"""
#%%

tests = os.listdir("./datasets/processed-2")

tests.pop(23)
tests.pop(16)

N = len(tests)

avg_max_a = 0
avg_half_sine = 0
for i, test in enumerate(tests):
    with open("./datasets/processed-2/" + test) as f:
        data_dict = json.load(f)
        f.close()
    a = np.array(data_dict['accelerometer_1'])
    t = np.array(data_dict['time'])
    max_a = np.max(a)
    argmax_a = np.argmax(a)
    # find half-sine time
    d = a > 1000
    # find starting point
    i0 = argmax_a
    while(d[i0]):
        i0 -= 1
    # find ending point
    i1 = argmax_a
    while(d[i1]):
        i1 += 1
    half_sine = t[i1] - t[i0]
    
    avg_max_a += max_a/N
    avg_half_sine += half_sine/N
    
    print(max_a)
    print(half_sine)
    
    d = np.logical_and(t>t[i0-1], t<t[i1+1])
    # plt.figure()
    # plt.plot(t[d], a[d], label='acceleration')
    # plt.plot([t[argmax_a]], [max_a], marker='o', linewidth=0, label='maximum acceleration')
    # plt.plot([t[i0], t[i1]], [a[i0], a[i1]], marker='o', linewidth=0, label='half sine points')
    # plt.legend()
    # plt.tight_layout()

print('max acceleration: ' + str(avg_max_a))
print('half sine: ' + str(avg_half_sine))
#%%