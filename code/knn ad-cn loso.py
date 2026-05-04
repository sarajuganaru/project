
# %%
import mne
import numpy as np
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

#%% open data
with open("params.yaml", "r") as file:
    params = yaml.full_load(file)

data = params['path_to_dataset']
derivatives = data  # path_to_dataset already points to derivatives
participants = os.path.join(os.path.dirname(data), 'participants.tsv')  # participants.tsv is in parent directory
# %% frequency bands
freq_bands = {
    'delta': (0.5, 4), #hpf
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 25),
    'gamma': (25, 45) #lpf of 45 
}

# %%
def process_subject(sub_id):
    """Calculates average relative power of all files"""
    # file set up
    file_name = f"{sub_id}_task-eyesclosed_eeg.set"
    file_path = os.path.join(derivatives, sub_id, 'eeg', file_name)
    if not os.path.exists(file_path):
        return None
    
    try:
        # read EEG w MNE
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        
        # calculates power spectral density with Welch (time to freq domain)
        psd_obj = raw.compute_psd(method='welch', fmin=0.5, fmax=45, verbose=False)
        
        psds = psd_obj.get_data() #pull numerical matrix (n_channels, n_freqs)
        freqs = psd_obj.freqs #array of actual freq values
        
        # calculates total power for each channel
        psds /= psds.sum(axis=-1, keepdims=True)
        
        # average power across all channels
        mean_psds = psds.mean(axis=0) 
    
        features = []
        # iterate through each freq band
        for fmin, fmax in freq_bands.values():
            # finds indeces of freq falling in bands
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            
            # average the power values within this specific frequency band
            features.append(mean_psds[idx].mean())
            
        # return the final feature vector as a numpy array 
        return np.array(features)

    except Exception as e:
        print(f"Error processing {sub_id}: {e}") #prints the error report
        return None #skips that subject and moves immediately to the next one
    
# %%  feature extraction

participants_df = pd.read_csv(participants, sep='\t')

# keep only A and C
participants_df = participants_df[participants_df['Group'].isin(['A', 'C'])]

X = [] #features
y = [] #labels
groups = [] # subject IDs

for index, row in participants_df.iterrows(): #goes through paricipant tables by row
    sub_id = row['participant_id']
    group = row['Group'] # A, C
    
    features = process_subject(sub_id)
    if features is not None:
        X.append(features)
        y.append(group)
        groups.append(sub_id)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)
groups = np.array(groups)

print(f"total processed subjects: {X.shape[0]}")

# %% check for NaN values
# find the indices of rows that contain NaN
nan_indices = np.where(np.isnan(X))[0] 

if len(nan_indices) > 0:
    for idx in nan_indices:
        print(f"Subject with NaN: {groups[idx]}") 
else:
    print("No NaN values found.")
# %% train knn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# pipeline: impute (NaN values), scale, knn
knn_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), #looks for NaN and replaces it with avg of that band across all other subjects
    ('scaler', StandardScaler()), #standardizes values
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
])

# Conduct LOSO
logo = LeaveOneGroupOut()
fold_accuracies = []
all_preds = []
all_true = []


# split into train/test
for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx] #extracts freq values using idx above
    y_train, y_test = y[train_idx], y[test_idx] #extracts label using idx above
     # train model
    knn_pipeline.fit(X_train, y_train)

    # predict
    y_pred = knn_pipeline.predict(X_test)

    # store predictions
    all_preds.extend(y_pred)
    all_true.extend(y_test)

    # compute fold accuracy
    fold_acc = accuracy_score(y_test, y_pred)
    fold_accuracies.append(fold_acc)


print(f"mean accuracy: {np.mean(fold_accuracies):.2f}")
# %% confusion matrix

cm = confusion_matrix(all_true, all_preds, labels=['A','C'])

print("\nConfusion Matrix:")
print(cm)


# %% plot confusion matrix

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['AD','Control']
)

disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# %% classification report

print("\nClassification Report:")
print(classification_report(
    all_true,
    all_preds,
    target_names=['AD','Control']
))


# %%
