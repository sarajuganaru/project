#%%
import numpy as np
import os
import yaml
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids
import pywt
from multiprocessing import Pool

#%% load params
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

bids_root  = os.path.dirname(params['path_to_dataset']) #bcs in bids participants.tsv is in the parent directory
deriv_root = params['path_to_dataset']
output_dir = params['scalogram_dir']
os.makedirs(output_dir, exist_ok=True) #create folder if it doesnt exist but prevents crashes if it is 

#%% load participant info
df = pd.read_csv(os.path.join(bids_root, 'participants.tsv'), sep='\t') #\t bcs columns are separated by tabs not commas
final_df = df[df['Group'].isin(['A', 'C'])].reset_index(drop=True) #df [] produces a mask and keeps the only ones where bool is true; in this case it would be true for A and C

#%% epoch params
e_len     = params['epoch_length']
e_overlap = e_len * params['overlap']

#%% define per-participant function
def process_participant(rdict):
    """ processes one participant from labels, eeg epoching, cwavelet to scalogram (per channel per epoch)"""
    s_id        = rdict['participant_id'].replace('sub-', '') #bids path already adds sub- 
    group       = rdict['Group']
    output_path = os.path.join(output_dir, f'sub-{s_id}.npy')

    if os.path.exists(output_path):
        print(f"subject {s_id} already exists")
        return {'participant_id': s_id, 'group': group, 'path': output_path} #so it is logged as processed despite skipping
    #try block to prevent full crash
    try:
        bp = BIDSPath(
            subject=s_id, 
            task='eyesclosed', 
            suffix='eeg',
            datatype='eeg', 
            root=deriv_root, 
            check=False
        )  #path defined
        raw = read_raw_bids(bp, verbose=False) #load path; verbose = false to prevent logging clutter
        if raw.get_montage() is None:
            raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
        raw.load_data() #eeg data now loaded into ram
        raw.resample(256.0)

        events = mne.make_fixed_length_events(raw, duration=e_len, overlap=e_overlap)
        epochs = mne.Epochs(
            raw, events, 
            tmin=0, 
            tmax=e_len - 1.0/256.0,
            baseline=None, #it is resting state so no need for baseline
            preload=True, 
            verbose=False
        )
        n_epochs   = len(epochs) #epoch count per participant
        ch_names   = epochs.ch_names #get channel names
        n_channels = len(ch_names) #get channel numbers
        n_times    = epochs.get_data().shape[-1] #time points is last in the epoch array

        dt           = 1.0 / 256.0
        wav_name     = 'cmor1.5-1.0'
        n_freqs      = 128
        target_freqs = np.logspace(np.log10(0.5), np.log10(45), n_freqs)
        scales       = pywt.central_frequency(wav_name) * 256.0 / target_freqs

        freq_path = os.path.join(output_dir, 'freqs.npy')
        if not os.path.exists(freq_path):
            np.save(freq_path, target_freqs) #we need the array for explaianbility later; better in npy so no rounding

        scalograms  = np.zeros((n_epochs, n_channels, n_freqs, n_times), dtype=np.float32)
        epochs_data = epochs.get_data()

        for ep_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                signal = epochs_data[ep_idx, ch_idx, :] #take 1 ep 1 ch all time pts
                coeffs, _ = pywt.cwt(signal, scales, wav_name, dt) # pywt returns both freqxtime coeff and target freqs but we already store them in .py
                scalograms[ep_idx, ch_idx] = np.abs(coeffs).astype(np.float32) ** 2

        np.save(output_path, scalograms)

        ch_path = os.path.join(output_dir, 'ch_names.txt') #we only had them in epochs as metadata but we now discard epochs and only keep scalograms; we still need them for interpretability
        if not os.path.exists(ch_path):
            with open(ch_path, 'w') as f:
                f.write('\n'.join(ch_names))

        print(f"subject {s_id} saved")
        return {'participant_id': s_id, 'group': group, 'path': output_path}

    except Exception as e:
        print(f"error processing subject {s_id}: {e}")
        return None  # failed participants return None

#%% run in parallel
if __name__ == '__main__':
    rows = final_df.to_dict(orient='records')  # converts final df to dict without keeping index labels

    with Pool(processes=4) as pool:
        results = pool.map(process_participant, rows)


    labels    = [r for r in results if r is not None] #filters out failed part
    labels_df = pd.DataFrame(labels) #turns the dict into table
    labels_df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False) #saves labels in csv without index
    print(f"\ndone {len(labels)} participants processed")
    print(labels_df['group'].value_counts().to_string())


