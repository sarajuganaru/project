#%%
import matplotlib.pyplot as plt
from mne.io import read_raw_eeglab
import numpy as np
import os
import yaml
from mne.io import read_raw_eeglab
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

# %% check if data is filtered bellow 0.5hz
with open("/Users/sarajuganaru/project/code/params.yaml", "r") as file:
    params = yaml.full_load(file)
file_path = os.path.join(params['path_to_dataset'], 'sub-002', 'eeg', 'sub-002_task-eyesclosed_eeg.set')
raw = read_raw_eeglab(file_path, preload=True)
psd = raw.compute_psd(
    fmin=0.01,
    fmax=10.0,
    n_fft=8192,
    n_jobs=1,
    verbose=False
)

freqs = psd.freqs
mean_psd = psd.get_data().mean(axis=0)

import matplotlib.pyplot as plt

plt.semilogy(freqs, mean_psd)
plt.axvline(0.5, color="red", linestyle="--", label="0.5 Hz")
plt.xlim(0.01, 10)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.legend()
plt.show()
# %%
