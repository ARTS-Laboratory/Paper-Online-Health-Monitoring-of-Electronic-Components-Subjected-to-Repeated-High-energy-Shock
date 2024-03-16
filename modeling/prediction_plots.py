import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.platform import tf_logging as logging
import os
import json
"""
figures from model predictions.
"""
plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif',
                                    'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
                                    'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
                                    'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.it': 'serif:italic'})
plt.rcParams.update({'mathtext.bf': 'serif:bold'})
plt.close('all')
#%%
""" maximum """
def maximum(signal, absolute=True):
    if(absolute):
        return max(np.max(signal), -np.max(-signal))
    else:
        return np.max(signal)
""" root mean squared """
def rms(signal):
    return np.sqrt(np.sum(np.square(signal))/signal.size)
""" standard deviation """
def std(signal, squared=False):
    m = np.mean(signal)
    s_ = np.sum(np.square(signal - m))
    s_ /= signal.size
    if(squared):
        return s_
    return np.sqrt(s_)
#%% process tests
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
    x_test = np.array(data_dict['accelerometer_2'])
    peak = np.argmax(x_test)
    x_test = x_test[peak-t_time_before:peak+t_time_after]
    x_test = x_test.reshape(1, -1, 1)
    
    if(X is None):
        X = x_test
    else:
        X = np.append(X, x_test, axis=0)
del data_dict
np.save('./datasets/X.npy', X)
# X = X - X[0]
xstd = np.std(X)
# normalize X
X = X/np.std(X)

# the first 10 tests are healthy, last 20 are unhealthy
y_healthy = np.full((15, X.shape[1], 2), [1, 0])
y_damaged = np.full((15, X.shape[1], 2), [0, 1])
y = np.append(y_healthy, y_damaged, axis=0)

# remove the middle ten tests to create a buffer in catagorization

X = np.append(X[:10,:,:], X[20:,:,:], axis=0)
y = np.append(y[:10,:,:], y[20:,:,:], axis=0)
#%% load model
model = keras.models.load_model('./model saves/health pred model')
#%% predict on dataset
for layer in model.layers:
    if(layer.stateful):
        layer.reset_states()
y_pred = model.predict(X)
np.save('./datasets/y_pred.npy', y_pred)
#%% figure
t = np.arange(0, 1300)*(10**-6)
l = 1100

t = t*10e6

for test in range(0, 20):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 3.5), sharex=True)
    ax1.plot([0, t[-1]], [.5, .5], '--', c='k')
    ax1.plot(t[:l], y_pred[test,:l,0])
    ax1.set_ylabel('health prediction')
    ax1.set_ylim([0, 1])
    ax1.set_yticks([.25, .75], ['unhealthy', 'healthy'])
    ax2.plot(t[:l], X[test,:l])
    ax2.set_xlim([0, t[l]])
    ax2.set_ylabel(r'acceleration (m/s$^2$)')
    ax2.set_xlabel(r'time (μs)')
    plt.tight_layout()
    plt.savefig('./figures/model predictions/test%d.png'%(test+1), dpi=500)
    plt.savefig('./figures/model predictions/test%d.svg'%(test+1))
#%% box and whisker plots for metrics
X = X.squeeze()
max_metric = np.max(X, axis=1)
rms_metric = np.sqrt(np.sum(np.square(X), axis=1))
std_metric = np.std(X, axis=1)

max_healthy = max_metric[:10]
max_unhealthy = max_metric[10:]
rms_healthy = rms_metric[:10]
rms_unhealthy = rms_metric[10:]
std_healthy = std_metric[:10]
std_unhealthy = std_metric[10:]

rr = [max_healthy, rms_healthy, std_healthy, max_unhealthy, rms_unhealthy, std_unhealthy]

plt.figure()
plt.plot([0]*10, max_healthy, linewidth=0, marker='.')
plt.plot([1]*10, max_unhealthy, linewidth=0, marker='.')

plt.figure()
plt.plot([0]*10, rms_healthy, linewidth=0, marker='.')
plt.plot([1]*10, rms_unhealthy, linewidth=0, marker='.')

plt.figure()
plt.plot([0]*10, std_healthy, linewidth=0, marker='.')
plt.plot([1]*10, std_unhealthy, linewidth=0, marker='.')


plt.figure()
for i in range(6):
    plt.plot([i]*10, rr[i])
#%% all experiments multiplot
y_pred = np.load('./datasets/y_pred.npy')
X = np.load('./datasets/X.npy')


fig, axes = plt.subplots(2, 10, sharex=True, sharey=True)

for i in range(2):
    for j in range(10):
        k = 2*j + i
        
        ax = axes[i,j]
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        y = y_pred[k][:,0]
        ax.plot(y)
fig.tight_layout()