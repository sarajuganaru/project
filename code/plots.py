#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
import mne
import numpy as np

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Load the data
df = pd.read_csv('/Users/sarajuganaru/project/dataset/ds004504/participants.tsv', sep='\t')

# Filter for Alzheimer's (A) and Control (C)
df_ac = df[df['Group'].isin(['A', 'C'])]

#%%
# Gender distribution by group
plt.figure(figsize=(8, 6))
sns.countplot(data=df_ac, x='Group', hue='Gender')
plt.title('Gender Distribution by Group (Alzheimer\'s vs Control)')
plt.xlabel('Group')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.savefig('output/gender_distribution.png')


#%%
# MMSE distribution
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_ac, x='Group', y='MMSE')
plt.title('MMSE Distribution by Group (Alzheimer\'s vs Control)')
plt.xlabel('Group')
plt.ylabel('MMSE Score')
plt.savefig('output/mmse_distribution.png')


#%%
# Age distribution
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_ac, x='Group', y='Age')
plt.title('Age Distribution by Group (Alzheimer\'s vs Control)')
plt.xlabel('Group')
plt.ylabel('Age')
plt.savefig('output/age_distribution.png')

#%%
# Age vs MMSE scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_ac, x='Age', y='MMSE', hue='Group')
plt.title('Age vs MMSE by Group')
plt.xlabel('Age')
plt.ylabel('MMSE Score')
plt.legend()
plt.savefig('output/age_mmse_scatter.png')

#%%
# Descriptive statistics
descriptives = df_ac.groupby('Group')[['Age', 'MMSE']].agg(['mean', 'std', 'count'])

# Create APA-style table
apa_table = "Group\tN\tAge\tMMSE\n"
apa_table += "-" * 40 + "\n"
for group in ['A', 'C']:
    group_name = 'Alzheimer\'s' if group == 'A' else 'Control'
    count = descriptives.loc[group, ('Age', 'count')]
    age_mean = descriptives.loc[group, ('Age', 'mean')]
    age_std = descriptives.loc[group, ('Age', 'std')]
    mmse_mean = descriptives.loc[group, ('MMSE', 'mean')]
    mmse_std = descriptives.loc[group, ('MMSE', 'std')]
    apa_table += f"{group_name}\t{int(count)}\t{age_mean:.2f} ({age_std:.2f})\t{mmse_mean:.2f} ({mmse_std:.2f})\n"

# Save APA-style descriptives to file
with open('output/descriptives_apa.txt', 'w') as f:
    f.write("Descriptive Statistics in APA Style\n")
    f.write(apa_table)
    f.write("\n")

# Also save the original format
with open('output/descriptives.txt', 'w') as f:
    f.write("Descriptive Statistics (Mean ± SD, Count)\n")
    f.write("=" * 50 + "\n")
    f.write(descriptives.round(2).to_string())
    f.write("\n\n")

# Plot descriptives table
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table_data = []
for group in ['A', 'C']:
    group_name = 'Alzheimer\'s' if group == 'A' else 'Control'
    count = descriptives.loc[group, ('Age', 'count')]
    age_mean = descriptives.loc[group, ('Age', 'mean')]
    age_std = descriptives.loc[group, ('Age', 'std')]
    mmse_mean = descriptives.loc[group, ('MMSE', 'mean')]
    mmse_std = descriptives.loc[group, ('MMSE', 'std')]
    table_data.append([group_name, str(int(count)), f"{age_mean:.2f} ({age_std:.2f})", f"{mmse_mean:.2f} ({mmse_std:.2f})"])
table = ax.table(cellText=table_data, colLabels=['Group', 'N', 'Age', 'MMSE'], loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Descriptive Statistics Table')
plt.savefig('output/descriptives_table.png')

# Gender counts
gender_counts = df_ac.groupby(['Group', 'Gender']).size().unstack(fill_value=0)

# Save gender counts to file
with open('output/gender_counts.txt', 'w') as f:
    f.write("Gender Counts by Group\n")
    f.write("=" * 30 + "\n")
    f.write(gender_counts.to_string())
    f.write("\n\n")

# Plot gender counts table
fig, ax = plt.subplots(figsize=(6, 3))
ax.axis('tight')
ax.axis('off')
table_data_gc = gender_counts.reset_index().values.tolist()
table_data_gc.insert(0, ['Group', 'F', 'M'])
table = ax.table(cellText=table_data_gc, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Gender Counts Table')
plt.savefig('output/gender_counts_table.png')
plt.close()

#%%
# Get recording durations
participants = pd.read_csv('/Users/sarajuganaru/project/dataset/ds004504/participants.tsv', sep='\t')
group_dict = dict(zip(participants['participant_id'], participants['Group']))
durations = []
for sub_dir in glob.glob('/Users/sarajuganaru/project/dataset/ds004504/sub-*/eeg/*.json'):
    with open(sub_dir, 'r') as f:
        data = json.load(f)
        duration = data.get('RecordingDuration', None)
        if duration:
            sub_id = sub_dir.split('/')[-3]  # sub-XXX
            group = group_dict.get(sub_id, None)
            if group in ['A', 'C']:
                durations.append({'Group': group, 'Duration': duration})
df_dur = pd.DataFrame(durations)

# Plot recording durations
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_dur, x='Group', y='Duration')
plt.title('Recording Durations by Group (Alzheimer\'s vs Control)')
plt.xlabel('Group')
plt.ylabel('Duration (seconds)')
plt.savefig('output/recording_durations.png')

#%%

# Compute duration statistics
dur_stats = df_dur.groupby('Group')['Duration'].agg(['mean', 'std', 'count'])

# Plot duration table
fig, ax = plt.subplots(figsize=(6, 3))
ax.axis('tight')
ax.axis('off')
table_data_dur = []
for group in ['A', 'C']:
    mean = dur_stats.loc[group, 'mean'] / 60  # Convert to minutes
    std = dur_stats.loc[group, 'std'] / 60    # Convert to minutes
    count = dur_stats.loc[group, 'count']
    group_name = 'Alzheimer\'s' if group == 'A' else 'Control'
    table_data_dur.append([group_name, f"{mean:.2f} ({std:.2f})", str(int(count))])
table = ax.table(cellText=table_data_dur, colLabels=['Group', 'Duration', 'N'], loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Recording Duration')
plt.savefig('output/duration_table.png')

#%%
# Bar plot of MMSE by Gender and Group
plt.figure(figsize=(8, 6))
sns.barplot(data=df_ac, x='Group', y='MMSE', hue='Gender')
plt.title('Mean MMSE by Group and Gender')
plt.xlabel('Group')
plt.ylabel('Mean MMSE Score')
plt.legend(title='Gender')
plt.savefig('output/mmse_gender_bar.png')
plt.close()

#%%
# Plot montage (no manual centering)
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
montage = mne.channels.make_standard_montage('standard_1020')
pos = montage.get_positions()['ch_pos']
selected_pos = {ch: pos[ch] for ch in ch_names if ch in pos}
landmarks = montage.get_positions()
montage_subset = mne.channels.make_dig_montage(
    ch_pos=selected_pos,
    nasion=landmarks['nasion'],
    lpa=landmarks['lpa'],
    rpa=landmarks['rpa'],
    coord_frame='head'
)
# Proportionally scale montage positions (smaller)
scale_factor = 0.8
scaled_pos = {ch: pos * scale_factor for ch, pos in selected_pos.items()}
scaled_landmarks = {
    'nasion': landmarks['nasion'] * scale_factor,
    'lpa': landmarks['lpa'] * scale_factor,
    'rpa': landmarks['rpa'] * scale_factor,
}
montage_subset_scaled = mne.channels.make_dig_montage(
    ch_pos=scaled_pos,
    nasion=scaled_landmarks['nasion'],
    lpa=scaled_landmarks['lpa'],
    rpa=scaled_landmarks['rpa'],
    coord_frame='head'
)
fig = mne.viz.plot_montage(
    montage_subset_scaled,
    show_names=True,
    show=False,
)
# Manually set font size for all text labels
for ax in fig.get_axes():
    for txt in ax.texts:
        txt.set_fontsize(13)
fig.tight_layout()

# %%
