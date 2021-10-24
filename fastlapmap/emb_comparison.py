# Fastlapmap and Sklearn Lap Eigenmap comparison

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from fastlapmap.spectral import LapEigenmap
from sklearn.datasets import load_digits

digits = load_digits()
data = digits.data

from scipy.sparse import csr_matrix
N_EIGS=2
N_NEIGHBORS=10
N_JOBS=10

sk_se = SpectralEmbedding(n_components=N_EIGS, n_neighbors=N_NEIGHBORS, n_jobs=N_JOBS).fit_transform(data)
flapmap_diff = LapEigenmap(data, n_eigs=N_EIGS, metric='euclidean', similarity='diffusion', norm_laplacian=True, k=N_NEIGHBORS, n_jobs=N_JOBS)
flapmap_fuzzy = LapEigenmap(data, n_eigs=N_EIGS, metric='euclidean', similarity='fuzzy', norm_laplacian=True, k=N_NEIGHBORS, n_jobs=N_JOBS)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Handwritten digits data:', fontsize=24)
ax1.scatter(sk_se[:, 0], sk_se[:, 1], c=digits.target, cmap='Spectral', s=5)
ax1.set_title('Sklearn\'s Laplacian Eigenmaps', fontsize=20)
ax2.scatter(flapmap_diff[:, 0], flapmap_diff[:, 1], c=digits.target, cmap='Spectral', s=5)
ax2.set_title('Fast Laplacian Eigenmaps with diffusion harmonics', fontsize=20)
ax3.scatter(flapmap_fuzzy[:, 0], flapmap_fuzzy[:, 1], c=digits.target, cmap='Spectral', s=5)
ax3.set_title('Fast Laplacian Eigenmaps with fuzzy simplicial sets', fontsize=20)
plt.show()

