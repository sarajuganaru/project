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

# Apply custom style
#plt.style.use("plot_style.mplstyle")


#%% Load params
with open("params.yaml", "r") as file:
    params = yaml.full_load(file)

#%%$$
root = os.path.dirname(params['path_to_dataset'])
data = read_raw_eeglab(os.path.join(params['path_to_dataset'],'sub-002', 'eeg','sub-002_task-eyesclosed_eeg.set'),preload=True)
data.describe()

# Plot to check the data for bad channels
data.plot(n_channels=19, duration=10, scalings='auto')


#%% check if .set has electrode locations
print(data.info['chs'][0])
print(data.get_montage())



#%% Extract temporal data
#mne.filter.filter_data(
#    data=data,

#)
eeg = data.get_data()
times = data.times


#%% Filter data
#sos = butter(4, 0.125, output='sos')
#y = sosfiltfilt(sos, x)

data.filter(l_freq=0.5, h_freq=45)

#%% Resample to 256 Hz
data.resample(256)


#%% Hilbert power


#%% Re-reference
data_avg_ref = data.copy().set_eeg_reference(ref_channels="average")

#%% ICA
data_for_ica = data.copy().filter(l_freq=1., h_freq=None)
ica = ICA(n_components=15, random_state=42, max_iter='auto')
ica.fit(data_for_ica)


#%% 
ica.plot_components()
ica.plot_sources(data)
ica.plot_properties(data, picks=[0,1])

#%% Heart Beats and blinks

# blinks
ica.plot_overlay(data, exclude=[0], picks="eeg")
#%% 
# Remove component 2

ica.exclude = [2]
data_clean = ica.apply(data.copy())

data_clean.plot()






#%% Plot
cmap = matplotlib.cm.get_cmap("viridis")
numChannels = 19
start_ts = 110000
end_ts = start_ts+1000 # add 2s
segment=np.arange(start_ts,end_ts)
plt.figure(figsize=(1,1))
for i in range(numChannels):
    plt.plot(data_avg_ref[i][1][segment],
             data_avg_ref[i][0].T[segment], 
             linewidth=.3,
             color=cmap(i/numChannels))
plt.xlabel('Time (s)')
plt.savefig('../../AD_patient_EEG_avg_ref_traces.pdf')

#%% Plot chatgpt
start_ts = 110000
end_ts = start_ts + 1000
segment = np.arange(start_ts, end_ts)

plt.figure(figsize=(10,6))

for i in range(19):
    plt.plot(
        data.times[segment],
        data.get_data()[i, segment] * 1e6 + i*100,
        linewidth=0.5
    )

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.show()



# # %% Embedd data
# embedding = UMAP(
#         n_components=2,
#         n_neighbors=30,
#         min_dist=.1,
#         metric='euclidean',
#         random_state=42
#         ).fit_transform(data.get_data(picks='all').T)
# # %%
# plt.scatter(embedding[:,0],
#             embedding[:,1],
#             s=.1,
#             marker='o',
#             c=data[0][1],
#             edgecolor='none',
#             alpha=.1)
# plt.axis('equal')
# plt.axis('off')
# # %%

# %%
