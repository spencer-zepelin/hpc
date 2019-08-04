import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import struct

n = 300

f = open("f0.bin", "rb")
x = np.fromfile(f, dtype=np.dtype('f8'))
x.shape = (n, n)
ax = sns.heatmap(x, xticklabels=50, yticklabels=50, vmin=0, vmax=1).set_title("At Initialization - Serial")
sns.plt.show()

sns.plt.clf()
f.close()

f = open("fb0.bin", "rb")
x = np.fromfile(f, dtype=np.dtype('f8'))
x.shape = (n, n)
ax1 = sns.heatmap(x, xticklabels=50, yticklabels=50, vmin=0, vmax=1).set_title("At Initialization - 36 MPI Ranks")
sns.plt.show()

sns.plt.clf()
f.close()

f = open("f1.bin", "rb")
x = np.fromfile(f, dtype=np.dtype('f8'))
x.shape = (n, n)
ax1 = sns.heatmap(x, xticklabels=50, yticklabels=50).set_title("Midway - Serial")
sns.plt.show()

sns.plt.clf()
f.close()

f = open("fb1.bin", "rb")
x = np.fromfile(f, dtype=np.dtype('f8'))
x.shape = (n, n)
ax1 = sns.heatmap(x, xticklabels=50, yticklabels=50).set_title("Midway - 36 MPI Ranks")
sns.plt.show()

sns.plt.clf()
f.close()

f = open("f2.bin", "rb")
x = np.fromfile(f, dtype=np.dtype('f8'))
x.shape = (n, n)
ax1 = sns.heatmap(x, xticklabels=50, yticklabels=50).set_title("At Termination - Serial")
sns.plt.show()

sns.plt.clf()
f.close()

f = open("fb2.bin", "rb")
x = np.fromfile(f, dtype=np.dtype('f8'))
x.shape = (n, n)
ax1 = sns.heatmap(x, xticklabels=50, yticklabels=50).set_title("At Termination - 36 MPI Ranks")
sns.plt.show()