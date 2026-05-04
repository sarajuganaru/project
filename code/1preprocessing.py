#%% 
import numpy as np
import os
import yaml
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids
import pywt


#%% load params
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

bids_root = os.path.dirname(params['path_to_dataset'])
deriv_root = params['path_to_dataset']
output_dir = params['scalogram_dir']
os.makedirs(output_dir, exist_ok=True) #to create if it doesnt exist

#%%load participant info
df = pd.read_csv(os.path.join(bids_root, 'participants.tsv'), sep='\t')
selected_df = df[df['Group'].isin(['A', 'C'])].reset_index(drop=True) #also resets index after filtering

#%%load epoch params
e_len = params['epoch_length']
e_overlap = e_len * params['overlap']

#%%participant processing 
labels = [] #this will have participant id, group, nyp path
for _, row in selected_df.iterrows():
    s_id = row['participant_id'].replace('sub-', '')
    group = row['Group']
    output_path = os.path.join(output_dir, f'sub-{s_id}.npy')

    if os.path.exists(output_path):
        print(f"Scalogram of subject {s_id} exists already so it will be skipped.")
        labels.append({'participant_id': s_id, 'group': group, 'path': output_path})
        continue
    try:
        # load EEG with mne-bids
        bp = BIDSPath(
            subject=s_id, task='eyesclosed', suffix='eeg', 
            datatype='eeg', root=deriv_root, check=False
        )
        raw = read_raw_bids(bp, verbose=False)
        raw.load_data()

        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        raw.resample(256.0)
        
        #epoching
        events = mne.make_fixed_length_events(raw, duration=e_len, overlap=e_overlap)
        epochs = mne.Epochs(
            raw, events, tmin=0, tmax=e_len - 1/256.0, 
            baseline=None, preload=True, verbose=False
        )
        n_epochs = len(epochs)
        ch_names = epochs.ch_names
        n_channels = len(ch_names)
        n_times = epochs.get_data().shape[-1]
        
        #wavelet parameters
        dt = 1.0 / 256.0
        wav_name = 'cmor1.5-1.0'
        n_freqs = 128
        target_freqs = np.logspace(np.log10(0.5), np.log10(45), n_freqs)
        scales = pywt.central_frequency(wav_name) * 256.0 / target_freqs
        #save freq info for later use
        freq_path = os.path.join(output_dir, 'freqs.npy')
        if not os.path.exists(freq_path):
            np.save(freq_path, target_freqs)

        #cwt scalograms
        scalograms = np.zeros((n_epochs, n_channels, n_freqs, n_times), dtype = np.float32) #output array
        
        #call get data once for all ch 
        epochs_data = epochs.get_data() #shape (n_epochs, n_channels, n_times)

        for ch_idx in range(n_channels):
            for epoch_idx in range(n_epochs):
                signal = epochs_data[epoch_idx, ch_idx, :]
                coeffs, _ = pywt.cwt(signal, scales, wav_name, dt)
                scalograms[epoch_idx, ch_idx] = np.abs(coeffs).astype(np.float32) **2

        np.save(output_path, scalograms)
        labels.append({'participant_id': s_id, 'group': group, 'path': output_path})
        print(f"Scalogram of subject {s_id} is saved successfully.")

        #save channel order
        ch_path = os.path.join(output_dir, 'ch_names.txt')
        if not os.path.exists(ch_path):
            with open(ch_path, 'w') as f:
                 f.write('\n'.join(ch_names))

    except Exception as e:
        print(f"Error processing subject {s_id}: {e}")



# %% save label csv
labels_df = pd.DataFrame(labels)
labels_df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)
print(f"\nDone. {len(labels_df)} participants processed.")
print(labels_df['group'].value_counts().to_string())
