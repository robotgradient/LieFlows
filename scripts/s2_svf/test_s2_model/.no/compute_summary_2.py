import numpy as np

N = 22

###################
m1 = np.array([0.902,0.1614,0.604,2.6501,0.9529,0.456,0.677,0.327,0.311,1.15,4.509,1.549,6.741,1.871,2.225,3.896,0.52,0.604,2.44,0.635,1.905,3.755])
a1 = np.array([0.52,0.351,0.2725,0.6798,0.881,0.621,0.2585,0.306,0.378,1.253,0.470,1.8593,0.582,0.31,0.501,0.9042,0.735,0.374,3.08,0.794,1.293,1.033])
d1 = np.array([29,59,31,42,42.5,36,35.5,10,40,36.5,27,37,16.5,19.5,58.5,21,27,16.5,45.5,22,55.5,19])
####################
m2 = np.array([1.205,0.051,0.84,4.152,2.45,0.295,2.64,0.206,0.1813,0.974,0.629,1.37,1.19,1.059,0.33,2.385,0.562,0.0765,0.739,0.157,0.661,3.82])
a2 = np.array([0.664,0.3437,0.509,2.639,4.61,0.442,0.3228,0.1407,0.2185,0.670,0.3410,1.953,2.88,0.5684,0.5041,1.9541,0.676,0.23,1.04,0.144,0.694,2.852])
d2 = np.array([26,0,0,59.5,33,0,0,0,31.5,4,4,78,0,25,22,21,67,8.5,75,43,0,0])
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

m = np.concatenate((m1[:],m2[:],m3[:]),0)
a = np.concatenate((a1[:],a2[:],a3[:]),0)
d = np.concatenate((d1[:],d2[:],d3[:]),0)
names = ['Kernel Coupling']*N + ['Coupling']*N + ['Ours']*N


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
plt.savefig('test_architecture.png', dpi=300)
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
plt.savefig('test_architecture2.png', dpi=300)
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
plt.savefig('test_architecture3.png', dpi=300)
plt.close(1)
