import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.fft import fft, fftfreq
"""
time and frequency domain multiplot
"""
#%%
plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.it': 'serif:italic'})
plt.rcParams.update({'mathtext.bf': 'serif:bold'})
plt.close('all')
#%%
test = os.listdir("./datasets/processed-2")[0]
dt = 1/51200
with open("./datasets/processed-2/" + test) as f:
    data_dict = json.load(f)
    f.close()

tt = np.array(data_dict['time']) * 1000
aa1 = np.array(data_dict['accelerometer_1'])
aa2 = np.array(data_dict['accelerometer_2'])

xf = fftfreq(aa1.size, dt)[:aa1.size//2]

ff1 = np.abs(fft(aa1)[:aa1.size//2])
ff2 = np.abs(fft(aa2)[:aa2.size//2])

# tt, aa1, aa2, freq, Z, theta = load_lvm(file_name + '.lvm')

# make a multiplot time series and impedance with frequency
pcb_center= tt[np.argmax(aa2)]
tt = tt - (pcb_center-1)

plt.figure(figsize=(6,6))
plt.subplot(211)
plt.plot(tt,aa1,'--d',lw=0.8,markersize=3,label='table')
plt.plot(tt,aa2,'-o',lw=0.8,markersize=3,label='PCB')
plt.xlim([0,5])
plt.ylim([-150000,200000])
plt.grid(True)
plt.xlabel('time (ms)')
plt.ylabel('acceleration (m/s$^2$)')
plt.legend()

ax2 = plt.subplot(212)
ax2.plot(xf/1000, ff1, label = 'table')
ax2.plot(xf/1000, ff2, label= 'PCB')
ax2.set_ylabel('frequency power (m/s$^2$)')
ax2.set_xlabel('frequency (kHz)')
ax2.grid(True)
ax2.set_xlim([0, 15])
plt.legend()
plt.ticklabel_format(style='plain')

plt.tight_layout()

plt.savefig('./figures/time and frequency.png', dpi=500)
plt.savefig('./figures/time and frequency.svg')
#%% just time domain plot
plt.figure(figsize=(6,3))
plt.plot(tt,aa1,'--d',lw=0.8,markersize=3,label='table')
plt.plot(tt,aa2,'-o',lw=0.8,markersize=3,label='PCB')
plt.xlim([0,5])
plt.ylim([-150000,200000])
plt.grid(True)
plt.xlabel('time (ms)')
plt.ylabel('acceleration (m/s$^2$)')
plt.legend()
plt.tight_layout()
plt.savefig('./figures/time.png', dpi=500)
plt.savefig('./figures/time.svg')
#%% just frequency domain
plt.figure(figsize=(6,3))
plt.plot(xf/1000, ff1, label = 'table')
plt.plot(xf/1000, ff2, label= 'PCB')
plt.ylabel('frequency power (m/s$^2$)')
plt.xlabel('frequency (kHz)')
plt.grid(True)
plt.xlim([0, 15])
plt.legend()
plt.ticklabel_format(style='plain')
plt.tight_layout()
plt.savefig('./figures/frequency.png', dpi=500)
plt.savefig('./figures/frequency.svg')
#%% impedance with impacts
tests = os.listdir("./processed-2")

tests.pop(23)
tests.pop(16)

z_ = np.zeros((5, len(tests)))

for i, test in enumerate(tests):
    name = './processed-2/' + test
    with open(name) as f:
        data_dict = json.load(f)
        f.close()
    
    z_[0, i] = data_dict['impedance 100 kHz'][0]
    z_[1, i] = data_dict['impedance 10 kHz'][0]
    z_[2, i] = data_dict['impedance 1 kHz'][0]
    z_[3, i] = data_dict['impedance 120 Hz'][0]
    z_[4, i] = data_dict['impedance 100 Hz'][0]


#%%

for i in range(5):
    print(np.mean(z_[i]))
    print(np.std(z_[i]))
    print()
    z_[i] = (z_[i] - np.mean(z_[i]))/np.std(z_[i])


plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(5, 3))
# plt.plot(range(1, len(tests)+1), z_[4], label='100 Hz')
# plt.plot(range(1, len(tests)+1), z_[3], label='120 Hz')
plt.plot(range(1, len(tests)+1), z_[2], label='1 kHz')
plt.plot(range(1, len(tests)+1), z_[1], label='10 kHz')
plt.plot(range(1, len(tests)+1), z_[0], label='100 kHz')
plt.xlabel('number of impacts')
plt.ylabel('normalized impedance magnitude')
plt.tight_layout()
plt.legend(loc=1)
plt.grid(True)
plt.savefig('./figures/impact impedance.png', dpi=500)
plt.savefig('./figures/impact impedance.svg')
#%%
# ax3 = ax2.twinx() # Create a twin of Axes with a shared x-axis but independent y-axis.
# ax3.set_xticks([0,1,2,3,4])
# ax2.set_xticklabels(freq)
# #ax3.set_xlabel(list(frequency))
# #ax1.get_shared_y_axes().join(ax1, ax3)
# #c1, = ax0.plot(x, np.sin(x), c='red')
# #c2, = ax1.plot(x, np.cos(x), c='blue')
# c3, = ax2.plot(Z,'-o',c=cc[2])
# c4, = ax3.plot(theta,'--d',c=cc[3])
# plt.legend([c3, c4], ["Z", "theta"])
# #loc = "upper left", bbox_to_anchor=(.070, 2.25))
# ax2.set_xlabel('frequency (Hz)')
# ax2.set_ylabel('Z (ohm)')
# ax3.set_ylabel('angle (degree)')
# ax2.set_ylim([-5e7,8e8])
# ax3.set_ylim([-92,-78])
#%% multiplot, impedance above features