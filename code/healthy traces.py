#%% Imports
import numpy as np
import os
import mne
from mne.io import read_raw_eeglab
from mne.preprocessing import ICA
import yaml
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt, butter, hilbert




#%% Load params
with open("params.yaml", "r") as file:
    params = yaml.full_load(file)

#%%
root = os.path.dirname(params['path_to_dataset'])
data = read_raw_eeglab(os.path.join(params['path_to_dataset'],'sub-064', 'eeg','sub-064_task-eyesclosed_eeg.set'),preload=True)
data.describe()

#%%
# Plot to check the data for bad channels
data.plot(n_channels=19, duration=10, scalings='auto')
# %%
