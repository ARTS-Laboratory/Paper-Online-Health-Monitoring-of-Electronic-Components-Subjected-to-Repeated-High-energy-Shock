import matplotlib.pyplot as plt
import numpy as np
import json
import os
import scipy
from numpy import ndarray
from scipy.signal import periodogram
"""
Extract features from the datasets
"""
#%%
plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.it': 'serif:italic'})
plt.rcParams.update({'mathtext.bf': 'serif:bold'})
plt.close('all')
#%% define all features
import numpy as np
# time-domain features
### kinetic energy related
""" maximum """
def maximum(signal, absolute=True):
    if(absolute):
        return max(np.max(signal), -np.max(-signal))
    else:
        return np.max(signal)
""" mean of absolute value """
def absolutemean(signal):
    return np.mean(np.abs(signal))
""" root mean squared """
def rms(signal):
    return np.sqrt(np.sum(np.square(signal))/signal.size)
### data statistics related
""" standard deviation """
def std(signal, squared=False):
    m = np.mean(signal)
    s_ = np.sum(np.square(signal - m))
    s_ /= signal.size
    if(squared):
        return s_
    return np.sqrt(s_)
""" skewness """
def skewness(signal):
    N = signal.size
    m = np.mean(signal)
    E = np.sum((signal-m)**3)
    return E/((N-1)*std(signal)**3)
""" kurtosis """
def kurtosis(signal):
    N = signal.size
    m = np.mean(signal)
    E = np.sum((signal-m)**4)
    return E/((N-1)*std(signal)**4)
### sinusoidal wave shape related
""" crest factor """
def crestfactor(signal):
    return maximum(signal)/rms(signal)
""" shape factor """
def shapefactor(signal):
    return rms(signal)/absolutemean(signal)
""" impulse factor """
def impulsefactor(signal):
    return maximum(signal)/absolutemean(signal)
#%%
def frequency_center(vector: ndarray, time_step: float) -> float:
    """ Return frequency center of vector."""
    frequencies, power = periodogram(vector, 1/time_step, scaling='spectrum')
    numerator = np.sum(power * frequencies, axis=-1)
    denominator = np.sum(power, axis=-1)
    return numerator / denominator


def root_mean_square_frequency(vector: ndarray, time_step: float) -> float:
    """ Return root mean square frequency."""
    frequencies, power_spectrum = periodogram(vector, 1/time_step, scaling='spectrum')
    return np.sqrt(np.sum(frequencies**2 * power_spectrum, axis=-1)
                   / np.sum(power_spectrum, axis=-1))


def root_variance_frequency(vector: ndarray, time_step: float) -> float:
    """ Return root variance frequency."""
    frequencies, power_spectrum = periodogram(vector, 1 / time_step, scaling='spectrum')
    if vector.ndim == 2:
        frequencies = np.expand_dims(frequencies, 0)
    freq_center = calc_frequency_center(power_spectrum, frequencies)
    if vector.ndim == 2:
        freq_center = np.expand_dims(freq_center, 1)
    return np.sqrt(np.sum(((frequencies - freq_center) ** 2) * power_spectrum, axis=-1)
                   / np.sum(power_spectrum, axis=-1))

def calc_frequency_center(
        power_spectrum: ndarray, frequencies: ndarray) -> float:
    """ Return frequency center of vector."""
    return np.sum(power_spectrum * frequencies, axis=-1) / np.sum(power_spectrum, axis=-1)
    # return np.mean(fft)


def calc_root_mean_square_frequency(
        power_spectrum: ndarray, frequencies: ndarray) -> float:
    """ Return root mean square frequency."""
    return np.sqrt(np.sum(power_spectrum * frequencies ** 2, axis=-1)
                   / np.sum(power_spectrum, axis=-1))
    # return np.mean(np.abs(np.asarray(fft)) ** 2)

def calc_root_variance_frequency(
        power_spectrum: ndarray, frequencies: ndarray) -> float:
    """ Return root variance frequency."""
    freq_center = calc_frequency_center(power_spectrum, frequencies)
    return np.sqrt(np.sum(((frequencies - freq_center) ** 2) * power_spectrum, axis=-1)
                   / np.sum(power_spectrum, axis=-1))
#%% time domain plots
# not including skewness
features = [maximum, absolutemean, rms, std, kurtosis, crestfactor,\
            shapefactor, impulsefactor]

tests = os.listdir("./datasets/processed-2")

tests.pop(23)
tests.pop(16)

feature_array = np.zeros([len(features), len(tests)])
for i, test in enumerate(tests):
    with open("./datasets/processed-2/" + test) as f:
        data_dict = json.load(f)
        f.close()
    s = np.array(data_dict['accelerometer_2'])
    for j, feature in enumerate(features):
        feature_array[j, i] = feature(s)

norm_features = np.copy(feature_array)

for i, feature in enumerate(norm_features):
    f1 = feature[0]
    for j, test in enumerate(feature):
        norm_features[i, j] = test/f1

norm_features = (norm_features - 1)*100

feature_names = ["max", "abs. mean", "RMS", "STD", "kurtosis", "crest factor",\
                  "shape factor", "impulse factor"]

plt.figure(figsize=(6, 4))
plt.xlabel("number of impacts")
plt.ylabel("percent change in metric value")
plt.xlim(1, len(tests))
plt.grid(True)
for i, feature in enumerate(norm_features):
    plt.plot(range(1, len(tests)+1), feature, label=feature_names[i], marker='.')
plt.legend(loc=2)
plt.tight_layout()
plt.savefig("./figures/time domain metrics.png", dpi=500)
plt.savefig("./figures/time domain metrics.svg")
#%% frequency domain plots
# not including skewness
features = [frequency_center, root_mean_square_frequency, root_variance_frequency]

tests = os.listdir("./datasets/processed-2")

tests.pop(23)
tests.pop(16)

dt = 1/51200
feature_array = np.zeros([len(features), len(tests)])
for i, test in enumerate(tests):
    with open("./datasets/processed-2/" + test) as f:
        data_dict = json.load(f)
        f.close()
    s = np.array(data_dict['accelerometer_2'])
    for j, feature in enumerate(features):
        feature_array[j, i] = feature(s, dt)

norm_features = np.copy(feature_array)

for i, feature in enumerate(norm_features):
    f1 = feature[0]
    for j, test in enumerate(feature):
        norm_features[i, j] = test/f1

norm_features = (norm_features - 1)*100

feature_names = ["freq. center", "RMS freq.", "root variance freq."]

plt.figure(figsize=(6, 4))
plt.xlabel("number of impacts")
plt.ylabel("percent change in metric value")
plt.xlim(1, len(tests))
plt.grid(True)
for i, feature in enumerate(norm_features):
    plt.plot(range(1, len(tests)+1), feature, label=feature_names[i], marker='.')
plt.legend(loc=2)
plt.tight_layout()
plt.savefig("./figures/freq domain metrics.png", dpi=500)
plt.savefig("./figures/freq domain metrics.svg")
#%% multiplot, impedance above features
features = [maximum, absolutemean, rms, std, kurtosis, crestfactor,\
            shapefactor, impulsefactor, frequency_center,\
            root_mean_square_frequency]

feature_names = ["max", "abs. mean", "RMS", "STD", "kurtosis", "crest factor",\
                  "shape factor", "impulse factor", "freq. center", "RMS freq."]

tests = os.listdir("./datasets/processed-2")

tests.pop(23)
tests.pop(16)

dt = 1/51200
feature_array = np.zeros([len(features), len(tests)])
z_ = np.zeros((5, len(tests)))
for i, test in enumerate(tests):
    with open("./datasets/processed-2/" + test) as f:
        data_dict = json.load(f)
        f.close()
    s = np.array(data_dict['accelerometer_2'])
    
    z_[0, i] = data_dict['impedance 100 kHz'][0]
    z_[1, i] = data_dict['impedance 10 kHz'][0]
    z_[2, i] = data_dict['impedance 1 kHz'][0]
    z_[3, i] = data_dict['impedance 120 Hz'][0]
    z_[4, i] = data_dict['impedance 100 Hz'][0]
    
    for j, feature in enumerate(features):
        if(j <= 7):
            feature_array[j, i] = feature(s)
        else:
            feature_array[j, i] = feature(s, dt)

for i in range(5):
    z_[i] = (z_[i] - np.min(z_[i]))/(np.max(z_[i]) - np.min(z_[i]))

norm_features = np.copy(feature_array)

for i, feature in enumerate(norm_features):
    f1 = feature[0]
    for j, test in enumerate(feature):
        norm_features[i, j] = test/f1

norm_features = (norm_features - 1)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4.85, 4))

ax1.plot(range(1, len(tests)+1), z_[2], label='1 kHz')
ax1.plot(range(1, len(tests)+1), z_[1], label='10 kHz')
ax1.plot(range(1, len(tests)+1), z_[0], label='100 kHz')
# ax1.set_xlabel('number of impacts')
ax1.set_ylabel('scaled impedance magnitude')
# ax1.legend(loc=1)
ax1.grid(True)

ax2.set_xlabel("number of impacts")
ax2.set_ylabel("change in metric value")
ax2.set_xlim(1, len(tests))
ax2.grid(True)
for i, feature in enumerate(norm_features):
    plt.plot(range(1, len(tests)+1), feature, label=feature_names[i], marker='.')
# ax2.legend(loc=2)
plt.tight_layout()
# fig.savefig("./plots/impedance_metrics.png", dpi=500)
fig.savefig("./figures/impedance_metrics.svg")