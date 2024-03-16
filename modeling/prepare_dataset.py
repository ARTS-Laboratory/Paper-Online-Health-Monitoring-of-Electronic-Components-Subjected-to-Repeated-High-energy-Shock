import numpy as np
import os
import json
import matplotlib.pyplot as plt
"""
Take the experimental data and prepare it for ML.
"""
t_time_before = 300 # number of samples taken before peak
t_time_after = 1000 # number of samples taken after peak
train_len = t_time_after + t_time_before

tests = os.listdir("./datasets/processed-2")

tests.pop(23)
tests.pop(16)

X = None

for i, test in enumerate(tests):
    with open("./datasets/processed-2/" + test) as f:
        data_dict = json.load(f)
        f.close()
    t_test = np.arange(train_len)/51200
    x_test = np.array(data_dict['accelerometer_2'])
    x_test_input = np.array(data_dict['accelerometer_1'])
    peak = np.argmax(x_test)
    x_test = x_test[peak-t_time_before:peak+t_time_after]
    x_test = x_test.reshape(1, -1, 1)
    x_test_input = x_test_input[peak-t_time_before:peak+t_time_after]
    x_test_input = x_test_input.reshape(1, -1, 1)
    
    # plt.figure()
    # plt.plot(t_test, x_test_input.flatten())
    # plt.plot(t_test, x_test.flatten())
    
    if(X is None):
        X = x_test
    else:
        X = np.append(X, x_test, axis=0)
del data_dict
# X = X - X[0]
# normalize X
X = X/np.std(X)

# the first 10 tests are healthy, last 20 are unhealthy
y_healthy = np.full((15, X.shape[1], 2), [1, 0])
y_damaged = np.full((15, X.shape[1], 2), [0, 1])
y = np.append(y_healthy, y_damaged, axis=0)

# remove the middle ten tests to create a buffer in catagorization

X = np.append(X[:10,:,:], X[20:,:,:], axis=0)
y = np.append(y[:10,:,:], y[20:,:,:], axis=0)

np.save('./datasets/X.npy', X)
np.save('./datasets/y.npy', y)