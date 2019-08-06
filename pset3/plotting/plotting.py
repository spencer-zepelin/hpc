import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

n = 1000

f = open("data.bin", "rb")
x = np.fromfile(f, dtype=np.int32)
x.shape = (n, n)
plt.figure(figsize=(15, 10))
ax = sns.heatmap(x, xticklabels=False, yticklabels=False, cbar=False).set_title("Julia - Serial", fontsize=40)
sns.plt.show()

sns.plt.clf()
f.close()

f = open("data_mpi_static.bin", "rb")
x = np.fromfile(f, dtype=np.int32)
x.shape = (n, n)
plt.figure(figsize=(15, 10))
ax = sns.heatmap(x, xticklabels=False, yticklabels=False, cbar=False).set_title("Julia - Static MPI", fontsize=40)
sns.plt.show()

sns.plt.clf()
f.close()

f = open("data_mpi_dynamic.bin", "rb")
x = np.fromfile(f, dtype=np.int32)
x.shape = (n, n)
plt.figure(figsize=(15, 10))
ax = sns.heatmap(x, xticklabels=False, yticklabels=False, cbar=False).set_title("Julia - Dynamic MPI", fontsize=40)
sns.plt.show()