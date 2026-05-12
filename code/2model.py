#%%
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold

#%%
class ScalogramDataset(Dataset):
    """
    loads cwt scalograms from npy files 
    shape of file is (n_epochs, n_channels, n_freqs, n_times) and function returns (n_channels, n_freqs, n_times) for each epoch and its lable
    """
    labels_map = {"A":1, "C":0}
    def __init__ (self, records, transform = None):
        self.records = records
        self.transform = transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        rec = self.records[index]
        data = np.load(rec['path'])
        epoch = data [rec['epoch_idx']]
        x = torch.tensor(epoch, dtype = torch.float32)
        if self.transform:
            x = self.transform (x)
        label = self.label_map[rec['group']]
        return x,torch.tensor(label, dtype = torch.long)

def normalize(x):
    """normalize scalogram using log transform and standardization"""
    x = torch.log1p(x)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x

def build_records (label_csv):
    """
    reads label and explands it into one record per epoch
    each record - dicr w path, group, part id, epoch idx
    """        
    df = pd.read_csv(label_csv)
    records = []
    for _, row in df.iterrows():
        #see how many epochs per entry
        data = np.load(row['path'])
        n_epochs = data.shape[0]  # n epochs is first from array
        for ep_idx in range(n_epochs):
            records.append({
                'path': row['path'],  # where npy file is
                'group': row['group'],
                'participant_id': row['participant_id'],
                'epoch_idx': ep_idx
            })
    return records     

def get_dataloaders(
        scalogram_dir,
        batch_size = 32,
        n_splits = 0,
        fold = 0,
        num_workers = 4,
        seed = 42,
):
    """
    makes train/val/test dataloaders w participant level stratified kfold
    """
    label_csv = os.path.join (scalogram_dir, 'labels.csv') 
    all_records = build_records (label_csv) #expand into one record per epoch
    participant_id = np.array([r['participant_id']for r in all_records])
    groups_str = np.array ([r['group'] for r in all_records])
    labels_int = np.array([ScalogramDataset.labels_map [g] for g in groups_str])
    sgkf = StratifiedGroupKFold (n_splits = n_splits, shuffle = True, random_state= seed)
    splits = list (sgkf.split(all_records, labels_int, groups=participant_id))
    train_idx, test_idx = splits [fold]
    #20% training set as validation set
    val_size = max (1, int (0.2 * len(train_idx)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_idx))
    val_idx= train_idx[perm[:val_size]]
    train_idx = train_idx[perm[val_size:]]

    #build 3 rec lists from idx arrays
    train_records = [all_records[i] for i in train_idx]
    val_records = [all_records[i] for i in val_idx]
    test_records = [all_records[i] for i in test_idx]


    #class balance
    train_labels = np.array ([ScalogramDataset.labels_map[r['group']]for r in train_records])
    class_counts = np.bincount(train_labels)

    #weight = 1/count so minority class gets samples more often
    class_weights = 1.0 / class_counts

    #assign a weight to every indiv training sample based on class
    sample_weights = torch.tensor([class_weights[l] for l in train_labels], dtype = torch.float)

    #now we use weights to oversample the minority class
    sampler = WeightedRandomSampler (sample_weights, num_samples = len (sample_weights), replacement = True)

    #create the 3 ds
    train_ds = ScalogramDataset(train_records, transform=normalize)
    val_ds = ScalogramDataset (val_records, transform = normalize)
    test_ds = ScalogramDataset (test_records, transform = normalize)

    #create the dataloader
    train_loader = DataLoader (train_ds, 
                            batch_size = batch_size,
                            sampler = sampler,
                            num_workers = num_workers, #for parallel data laoding to speed up
                            pin_memory = True) #speeds up
    val_loader = DataLoader (val_ds,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = num_workers,
                            pin_memory = True )
    test_loader = DataLoader(test_ds,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = num_workers,
                            pin_memory = True)

    return train_loader, val_loader, test_loader

