#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import pywt
import yaml
import os
from mne.io import read_raw_eeglab
from scipy.io import loadmat

#%% load params
with open("params.yaml", "r") as file:
    params = yaml.full_load(file)
#%% Load data
data = read_raw_eeglab(os.path.join(params['path_to_dataset'],'derivatives','sub-002', 'eeg','sub-002_task-eyesclosed_eeg.set'))
Fs = params['sampling_frequency']
dt=1/Fs
#%% extract only one channel to form 1D array
channel_data = data.get_data(picks=1)[0] 
#%% Compute wavelet
%%time
wav_name = 'cmor1-1.5'
scales = np.logspace(np.log10(8),np.log10(512), 128)

frequencies = pywt.scale2frequency(wav_name, scales)/dt

coefs, freqs = pywt.cwt(
    channel_data,
    scales=scales,
    wavelet=wav_name,
    sampling_period=dt
    )

total_absolute_energy = abs(coefs)**2

#%% OLD Plot spectrum
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

plt.savefig('../../wavelet.pdf')

#%% New plot spectrum
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 7))

# 1. Flip the data if necessary
# If your frequencies array goes from 8Hz (index 0) to 512Hz (index 511),
# but you want 512Hz at the top, you need to use origin='upper'.
plt.imshow(total_absolute_energy, 
           cmap='magma', 
           aspect='auto', 
           origin='upper', # High frequencies usually go at the top
           rasterized=True)

# 2. Fix the Logarithmic Y-axis labels
# Instead of linspace, we pick indices that represent the Hz we care about
target_hz = [1, 4, 8, 12, 30, 60, 100] # Standard EEG bands
ytick_indices = []
ytick_labels = []

for hz in target_hz:
    # Find the index in your 'frequencies' array closest to the target Hz
    idx = (np.abs(frequencies - hz)).argmin()
    ytick_indices.append(idx)
    ytick_labels.append(f"{hz} Hz")

plt.yticks(ytick_indices, ytick_labels)
plt.ylabel('Frequency (Hz) - Log Scale')

# 3. Time Axis (X)
num_samples = channel_data.shape[0]
total_seconds = num_samples / Fs
xtick_indices = np.linspace(0, num_samples, 8)
xtick_labels = [f"{t:.0f}" for t in np.linspace(0, total_seconds, 8)]
plt.xticks(xtick_indices, xtick_labels)
plt.xlabel('Time (s)')

# 4. Better Contrast Control
# If it's too dark, lower the percentile (e.g., 90)
vmax_val = np.percentile(total_absolute_energy, 95) 
plt.clim(0, vmax_val)

plt.title('Readable EEG Scalogram: Subject 002')
plt.colorbar(label='Absolute Energy')
plt.show()


# %%

plt.imshow(total_absolute_energy, 
           aspect='auto', 
           cmap='viridis', 
           origin='lower',
           interpolation='bilinear')


target_hz = [1, 4, 8, 12, 20, 30, 40]
ytick_indices = [(np.abs(frequencies - hz)).argmin() for hz in target_hz]
plt.yticks(ytick_indices, [f"{hz}Hz" for hz in target_hz])


plt.xticks(np.linspace(0, total_absolute_energy.shape[1], 6), 
           np.linspace(0, 5, 6))

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'High-Resolution Scalogram ({wav_name})')
plt.colorbar(label='Absolute Energy')


vmax_val = np.percentile(total_absolute_energy, 99)
plt.clim(0, vmax_val)

plt.show()
# %%
