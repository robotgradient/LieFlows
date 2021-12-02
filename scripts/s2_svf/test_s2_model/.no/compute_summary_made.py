import numpy as np

N = 22


####################
m2 = np.array([3.679,0.019,0.012,0.01517,0.014,0.3203,0.623,0.606,0.411,1.176,1.56,0.033,1.782,0.064,0.037,0.347,0.887,0.1961,1.953,1.3082,2.32, 5.63])
a2 = np.array([1.827,0.121, 1.0168,0.09,1.047,0.054,0.215,1.225,1.158,0.0758,1.813, 1.16,0.0527,1.198,0.0482,0.946,0.919,0.2619,3.908,0.2814,0.998,1.075])
d2 = np.zeros(N)
####################
m3 = np.array([0.679,0.0019,0.012,0.1517,0.014,0.5203,0.623,0.606,0.311,1.176,2.56,2.033,0.782,1.064,0.937,2.347,0.887,0.1961,0.953,1.3082,1.32,4.63])
a3 = np.array([0.827,0.021,0.0168,0.9,0.047,0.54,0.215,0.225,0.158,0.758,0.813,2.16,0.527,0.198,0.482,0.946,0.919,0.2619,2.908,0.2814,0.998,1.075])
d3 = np.zeros(N)

print(np.sum(m3)/N)
print(np.sum(a3)/N)
print(np.sum(d3)/N)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from pylab import *

m = np.concatenate((m2[:],m3[:]),0)
a = np.concatenate((a2[:],a3[:]),0)
d = np.concatenate((d2[:],d3[:]),0)
names = ['MADE']*N + ['SoftMADE']*N


df2 = pd.DataFrame({
    "Layer": names,
    "MSE": m,
    "Area": a,
    "Instability %" : d
})

# tips = sns.load_dataset("tips")
rc('axes', linewidth=4)
rc('font', weight='bold')
rc('xtick', labelsize=20)

plt.figure(1, figsize=(9, 7))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

sns.set_style('whitegrid')
sns.set_context("poster", rc={"font.weight":'bold', "axes.titlesize":8, "axes.labelsize":30,
                              "xtick.labelsize":25, "ytick.labelsize":30, "ytick.weight":"bold",  "xtick.weight":"bold", "axes.labelweight":"bold"})

sns.violinplot(x="Layer", y="MSE", data=df2, cut=0)
plt.tight_layout()
plt.savefig('test_architecture_made_1.png', dpi=300)
plt.close(1)
##################################################


plt.figure(1, figsize=(9, 7))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

sns.set_style('whitegrid')
sns.set_context("poster", rc={"font.weight":'bold', "axes.titlesize":8, "axes.labelsize":30,
                              "xtick.labelsize":25, "ytick.labelsize":30, "ytick.weight":"bold",  "xtick.weight":"bold", "axes.labelweight":"bold"})

sns.violinplot(x="Layer", y="Area", data=df2, cut=0)
plt.tight_layout()
plt.savefig('test_architecture_made_2.png', dpi=300)
plt.close(1)
##################################################
plt.figure(1, figsize=(10, 7))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

sns.set_style('whitegrid')
sns.set_context("poster", rc={"font.weight":'bold', "axes.titlesize":8, "axes.labelsize":30,
                              "xtick.labelsize":25, "ytick.labelsize":30, "ytick.weight":"bold",  "xtick.weight":"bold", "axes.labelweight":"bold"})


sns.violinplot(x="Layer", y="Instability %", data=df2, cut=0)
plt.tight_layout()
plt.savefig('test_architecture_made_3.png', dpi=300)
plt.close(1)
