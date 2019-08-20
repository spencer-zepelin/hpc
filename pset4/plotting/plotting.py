import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.cm as cm

# Grid dimension
n = 1000

f = open("data.bin", "rb")
x = np.fromfile(f, dtype=np.float64)
x.shape = (n, n)
plt.figure(figsize=(8, 8))
ax = sns.heatmap(x, xticklabels=False, yticklabels=False, cbar=False, cmap='binary').set_title("Sample Image", fontsize=24)
sns.plt.show()

sns.plt.clf()
f.close()

