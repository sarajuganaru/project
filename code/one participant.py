#%%
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import yaml
import mne
from mne.io import read_raw_eeglab

#%% load parameters
with open("/Users/sarajuganaru/project/code/params.yaml", "r") as file:
    params = yaml.full_load(file)

# look at sub 002
file_path = os.path.join(params['path_to_dataset'], 'sub-002', 'eeg', 'sub-002_task-eyesclosed_eeg.set')
raw = read_raw_eeglab(file_path, preload=True)


# downsample to 256hz
raw.resample(256.0)
params['sfreq'] = 256

# write sample frequency
Fs = params['sfreq'] 
dt = 1 / Fs

#%%
raw.get_channel_types()
#%%
raw.get_montage()
#%%
raw.info['chs']


#%% create 2s epoch w 50% overlap
events = mne.make_fixed_length_events(raw, duration=params['epoch_length'])
epochs = mne.Epochs(raw, events, tmin=0, tmax=params['epoch_length'], 
                    baseline=None, preload=True, verbose=False)

#%% wavelet parameters
wav_name = 'cmor1.5-1.0' 
target_freqs = np.linspace(0.5, 45, 128)
scales = pywt.central_frequency(wav_name) * Fs / target_freqs
frequencies = pywt.scale2frequency(wav_name, scales) * Fs

#%% compute scalogram for first epoch of O1 channel 
target_channel = 'O1'
# bcs epochs.get_data() is (n_epochs, n_channels, n_times), select first epoch
first_epoch_data = epochs.get_data(picks=target_channel)[46, 0, :] 

# compute CWT
coefs, _ = pywt.cwt(first_epoch_data, scales, wav_name, sampling_period=dt)
power = np.abs(coefs)**2


#%% plot
import matplotlib.colors as colors

fig, ax = plt.subplots(figsize=(10, 6))

# using log norm
pcm = ax.pcolormesh(epochs.times, frequencies, power, 
                   shading='gouraud', cmap='jet',
                   norm=colors.LogNorm(vmin=power.min() + 1e-12, vmax=power.max()))

ax.set_yscale('log', base=2)
ax.set_ylim(0.5, 45)
ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32, 45])
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_title(f"Scalogram: Subject 002 - {target_channel} (Epoch 47)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")

fig.colorbar(pcm, ax=ax, label='Power (μV²/Hz)')
plt.tight_layout()
plt.show()


# %%
