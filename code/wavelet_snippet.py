#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import yaml
import mne
from mne.io import read_raw_eeglab
from scipy.io import loadmat

# Apply custom style
#plt.style.use("plot_style.mplstyle")

#%% Load LFP data (sleep) from Matlab file
with open("/Users/sarajuganaru/project/code/params.yaml", "r") as file:
    params = yaml.full_load(file)

data = read_raw_eeglab(os.path.join(params['path_to_dataset'],'sub-002', 'eeg','sub-002_task-eyesclosed_eeg.set'),preload=True) #load your data here
Fs = 500 # you have to know what is your sampling frequency
#Print


#%% Compute wavelet
dt = 1 / Fs
wav_name = 'cmor1-1.5'

# Reduce number of scales to speed up computation
scales = np.logspace(np.log10(8), np.log10(512), 128)  # 128 instead of 512

frequencies = pywt.scale2frequency(wav_name, scales) / dt

# Extract only the first channel
channel_data = data.get_data(picks=1)[0]  # 1D array

# Compute CWT for the first channel
coefs, freqs = pywt.cwt(
    channel_data,
    scales=scales,
    wavelet=wav_name,
    sampling_period=dt
)

total_absolute_energy = abs(coefs)**2


#%% Plot spectrum
#plt.xlim(5.121*1e6,5.125*1e6)
plt.axis('off')

plt.subplot(212)
plt.imshow(total_absolute_energy,
           cmap='magma',
           vmin=0,
           vmax=.0000008,
           aspect='auto',
           rasterized=True)
plt.yticks([0,len(frequencies)],[200,2])
plt.ylabel('Frequency (Hz)')
plt.xticks([50,2050,4050],[-2,0,2])
plt.xlabel('Time from IED (s)')

plt.savefig('/Users/sarajuganaru/project/output')

# %%

import matplotlib
# Use Agg backend to avoid GUI issues on Mac
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Assume total_absolute_energy and frequencies already computed for one channel
# channel_data is your 1D EEG array
time = np.arange(channel_data.size) / Fs  # time axis in seconds

plt.figure(figsize=(12, 5))

# Plot wavelet power spectrum
plt.imshow(
    total_absolute_energy,
    aspect='auto',
    cmap='magma',
    extent=[time[0], time[-1], frequencies[-1], frequencies[0]],  # time x freq
    vmin=0,
    vmax=np.max(total_absolute_energy)
)

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Wavelet Power Spectrum (Channel 1)')
plt.colorbar(label='Power')

# Save the figure without opening a window
plt.savefig('wavelet_power_channel1.png', dpi=150, bbox_inches='tight')
#plt.close()
plt.show()

# %%
