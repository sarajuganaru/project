#%% Imports
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pingouin as pg

# Apply custom style
#plt.style.use("plot_style.mplstyle")

#%% Greek dataset:
df = pd.read_csv(
    '../dataset/ds004504/participants.tsv',
    sep='\t'             )

#%%
plt.figure(figsize=(1,.5))
sns.stripplot(
    data=df.query("Group=='A' or Group=='C'"),
    y='MMSE',
    x='Group',
    hue='Gender',
    palette=['gray','gray'],
    size=1,
    dodge=True,
    legend=False
    
)
sns.barplot(
    data=df.query("Group=='A' or Group=='C'"),
    y='MMSE',
    x='Group',
    hue='Gender',
    palette=['C3','C0'],
    errorbar='se',
    capsize=.2,
    legend=False
)

plt.xticks([0,1],['AD', 'Healthy'])
plt.xlabel('')
plt.savefig('../output/MMSE_AD_male_female.pdf')

#%% STATS
anova_results = pg.anova(
    data=df.query("Group=='A' or Group=='C'"),
    dv='MMSE',
    between=['Group','Gender']
).round(4)
print(anova_results)
anova_results.to_csv('../output/MMSE_ANOVA_Group_Gender.csv')




#%% Load longitudinal dataset
df = pd.read_csv('../../datasets/AD_IEDs_longitudinal/AD_patients_IEDs_MMSE_longitudinal.csv')
# %%
plt.figure(figsize=(1,.5))
sns.lineplot(
    data=df,
    x='Time (Months)',
    y='MMSE',
    hue='IEDs',
    size=.3,
    alpha=.5,
    palette=['C6','C0'],
    estimator=None,
    units='Patient ID',
    legend=False
)

sns.regplot(
    data=df.query("IEDs=='No'"),
    x='Time (Months)',
    y='MMSE',
    color='C0',
    marker=''
)
sns.regplot(
    data=df.query("IEDs=='Yes'"),
    x='Time (Months)',
    y='MMSE',
    color='C6',
    marker=''
)
plt.xlabel('Time since first\nvisit (months)')
plt.ylim(0,30)
plt.savefig('../../MMSE_IEDs_AD_longitudinal.pdf')
# %%
# %%
plt.figure(figsize=(1,.5))
sns.lineplot(
    data=df,
    x='Time (Months)',
    y='MMSE',
    size=.3,
    alpha=.5,
    color='C6',
    estimator=None,
    units='Patient ID',
    legend=False
)

sns.regplot(
    data=df,
    x='Time (Months)',
    y='MMSE',
    color='C6',
    marker=''
)
plt.xlabel('Time since first\nvisit (months)')
plt.ylim(0,30)
plt.savefig('../../MMSE_AD_longitudinal.pdf')
# %%
