import numpy as np



N = 22

m1 = np.array([0.004 ,0.027,0.028,0.042,0.015,0.012,0.044,0.009,0.007,0.005,0.029,0.012,0.053,0.0202,0.0559,0.108,0.0131,0.007,0.0154,0.0304,0.1151,0.01])
print(np.sum(m1)/N)
a1 = np.array([0.02,0.01,0.052,0.099,0.102,0.202,0.0378,0.0383,0.0151,0.1561,0.0447,0.17,0.116,0.0552,0.0531,0.1229,0.0601, 0.046,0.0484,0.1629,0.0134,0.110])
print(np.sum(a1)/N)
d1 = np.array([0.008,0.003,0.023,0.011,0.013,0.087,0.087, 0.0076,0.003,0.0034,0.022,0.0064,0.076,0.020,0.0089,0.0092,0.021,0.0042,0.0133,0.0143,0.011,0.0082,0.0157])
print(np.sum(d1)/N)

####################
m2 = np.array([0.03,0.051,0.03,0.086,0.054,0.038,0.00228,0.037,0.051,0.0218,0.051,0.0217,0.051, 0.0129,0.09,0.097,0.082,0.082,0.0865,0.044,0.0146,0.167])

a2 = np.array([0.11,0.08,0.0949,0.133,0.146,0.115,0.02,0.0921,0.1836,0.358,0.066826,0.379,0.14,0.20,0.0954, 0.41,0.0639,0.195,0.45,0.081,0.25,0.29])
d2 = np.array([0.003,0.01,0.045,0.0717,0.011,0.016,0.0098,0.042,0.132,0.084,0.0119,0.027,0.03,0.051,0.052,0.0215,0.0119,0.092,0.0201,0.0383, 0.0389])

print(np.sum(m2)/N)
print(np.sum(a2)/N)
print(np.sum(d2)/N)


####################
m3 = np.array([0.00076, 0.0019,0.012,0.01517,0.014,0.0586, 0.0232,0.023,0.023,0.012,0.027,0.062,0.013,0.092,0.00304,0.011,0.0021,0.0123,0.0227,0.00308,0.00357,0.0189])
a3 = np.array([0.0071,0.021,0.0168,0.09, 0.047,0.0474,0.0171,0.01907,0.0683, 0.01077,0.233,0.0423, 0.0269, 0.03904, 0.017, 0.0619,0.0285, 0.2512, 0.0226, 0.00267, 0.054, 0.07])
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
names = ['Eucl-Flows']*N + ['IFlows']*N + ['S2-StableFlows']*N


df2 = pd.DataFrame({
    "Method": names,
    "MSE": m,
    "Area": a,
    "S2 deviation" : d
})

# tips = sns.load_dataset("tips")
#rc('text', usetex=True)
rc('axes', linewidth=4)
rc('font', weight='bold')
rc('xtick', labelsize=20)

plt.figure(1, figsize=(9, 7))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

sns.set_style('whitegrid')
sns.set_context("poster", rc={"font.weight":'bold', "axes.titlesize":8, "axes.labelsize":30,
                              "xtick.labelsize":25, "ytick.labelsize":30, "ytick.weight":"bold",  "xtick.weight":"bold", "axes.labelweight":"bold"})

sns.violinplot(x="Method", y="MSE", data=df2, cut=0)
plt.tight_layout()
plt.savefig('extended_experiment.png', dpi=300)
plt.close(1)

############################
rc('axes', linewidth=4)
rc('font', weight='bold')
plt.figure(1, figsize=(9, 7))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
sns.set_style('whitegrid')
sns.set_context("poster", rc={"font.weight":'bold', "axes.titlesize":8, "axes.labelsize":30,
                              "xtick.labelsize":25, "ytick.labelsize":30, "ytick.weight":"bold",  "xtick.weight":"bold", "axes.labelweight":"bold"})

sns.violinplot(x="Method", y="Area", data=df2, cut=0)
plt.tight_layout()
plt.savefig('test2.png', dpi=300)
plt.close(1)


####################
rc('axes', linewidth=4)
rc('font', weight='bold')
plt.figure(1, figsize=(9, 7))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
sns.set_style('whitegrid')
sns.set_context("poster", rc={"font.weight":'bold', "axes.titlesize":8, "axes.labelsize":30,
                              "xtick.labelsize":25, "ytick.labelsize":30, "ytick.weight":"bold",  "xtick.weight":"bold", "axes.labelweight":"bold"})
sns.violinplot(x="Method", y="S2 deviation", data=df2, cut=0)
plt.tight_layout()
plt.savefig('test3.png', dpi=300)
plt.close(1)
