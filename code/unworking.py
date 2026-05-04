ll_epochs = []

#%% epoch
for _, row in selected_df.iterrows():
    s_id = row['participant_id'].replace('sub-', '')
    
    bp = BIDSPath(
        subject=s_id, task='eyesclosed', suffix='eeg', 
        datatype='eeg', root=deriv_root, check=False
    )
    
    try:
        # load derivative
        raw = read_raw_bids(bp, verbose=False)
        raw.load_data()
        
        # add channel types & montage
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
        # resample
        raw.resample(256.0)
        
        # overlap epochs
        e_len = params['epoch_length']
        e_overlap = e_len * params['overlap']
        events = mne.make_fixed_length_events(raw, duration=e_len, overlap=e_overlap)
        
        epochs = mne.Epochs(
            raw, events, tmin=0, tmax=e_len, 
            baseline=None, preload=True, verbose=False
        )
        
        # attach group label to metadta
        epochs.metadata = pd.DataFrame({'Group': [row['Group']] * len(epochs)})
        
        all_epochs.append(epochs)
        print(f"Processed sub-{s_id} ({row['Group']})")
        
    except Exception as e:
        print(f"Error with sub-{s_id}: {e}")

#%% combine all epochs
if all_epochs:
    combined_epochs = mne.concatenate_epochs(all_epochs)
    print(f"\nFinal Dataset: {len(combined_epochs)} total epochs.")

# %% check if group labels stayed
print(combined_epochs.metadata['Group'].value_counts())

# %% wavelet analysis for scalograms
import pywt
import matplotlib.colors as colors

# get Fs and dt from yaml params
Fs = params['sfreq']
dt = 1 / Fs

# wavelet parameters
wav_name = 'cmor1.5-1.0' 
target_freqs = np.logspace(np.log10(0.5), np.log10(45), 128)
scales = pywt.central_frequency(wav_name) * Fs / target_freqs


# compute scalogram for each epoch of all channels
all_channels = combined_epochs.ch_names  # list of all channel names

for ch_idx, target_channel in enumerate(all_channels):
    epoch_data = combined_epochs.get_data(picks=target_channel)  # shape: (n_epochs, 1, n_times)
    
    for epoch_idx in range(len(combined_epochs)):
        epoch_signal = epoch_data[epoch_idx, 0, :]
        
        # compute CWT
        coefs, _ = pywt.cwt(epoch_signal, scales, wav_name, sampling_period=dt)
        power = np.abs(coefs)**2
        
        # plot
        fig, ax = plt.subplots(figsize=(10, 6))
        pcm = ax.pcolormesh(combined_epochs.times, frequencies, power, 
                           shading='gouraud', cmap='jet',
                           norm=colors.LogNorm(vmin=power.min() + 1e-12, vmax=power.max()))
        ax.set_yscale('log', base=2)
        ax.set_ylim(0.5, 45)
        ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32, 45])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_title(f"Scalogram: {target_channel} (Epoch {epoch_idx})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(pcm, ax=ax, label='Power (μV²/Hz)')
        plt.tight_layout()
        fig.savefig(f'output/scalogram_{target_channel}_epoch_{epoch_idx:03d}.png')
        plt.close(fig)

print("Scalograms saved for all epochs and all channels.")