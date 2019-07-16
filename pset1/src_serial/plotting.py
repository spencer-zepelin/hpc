import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

f = open('f0.txt', 'r')
x = np.loadtxt(f)
ax = sns.heatmap(x, xticklabels=50, yticklabels=50, vmin=0, vmax=1).set_title("At Initialization")
sns.plt.show()
sns.plt.clf()
f.close()

f = open('f1.txt', 'r')
x = np.loadtxt(f)
ax1 = sns.heatmap(x, xticklabels=50, yticklabels=50, vmin=0, vmax=1).set_title("Halfway Point")
sns.plt.show()
sns.plt.clf()
f.close()

f = open('f2.txt', 'r')
x = np.loadtxt(f)
ax1 = sns.heatmap(x, xticklabels=50, yticklabels=50, vmin=0, vmax=1).set_title("At Completion")
sns.plt.show()
sns.plt.clf()
f.close()