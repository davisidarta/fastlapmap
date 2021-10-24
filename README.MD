[![Latest PyPI version](https://img.shields.io/pypi/v/fastlapmap.svg)](https://pypi.org/project/fastlapmap/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/DaviSidarta.svg?label=Follow%20%40davisidarta&style=social)](https://twitter.com/davisidarta)
        
# Fast Laplacian Eigenmaps in python

Open-source Laplacian Eigenmaps for dimensionality reduction of large data in python. Comes with
 approximate-nearest-neighbors wrappers for [NMSlib](https://github.com/nmslib/nmslib) and [HNSWlib](https://github.com/nmslib/hnswlib).

Performs around 12x faster than the default [scikit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html).    
        
# Installation

You'll need NMSlib and HNSWlib for using this package, along with some basic libraries:

```
pip3 install numpy pandas scikit-learn nmslib hnswlib
```

Then you can install it with pip:

```
pip3 install fastlapmap
```


# Usage 

See the following example with the [handwritten digits data](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits).
Here, I visually compare results from the scikit-learn Laplacian Eigenmaps 
[implementation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html#sklearn.manifold.SpectralEmbedding) to 
those from my implementation.

Note that this implementation contains two similarity-learning algorithms: [anisotropic diffusion maps](https://doi.org/10.1073/pnas.0500334102) and [fuzzy simplicial sets](https://arxiv.org/abs/1802.03426).


```
# Import libraries
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

flapmap_diff = LapEigenmap(data, n_eigs=2, metric='euclidean', similarity='diffusion', norm_laplacian=True, k=N_NEIGHBORS, n_jobs=N_JOBS)
flapmap_fuzzy = LapEigenmap(data, n_eigs=2, metric='euclidean', similarity='fuzzy', norm_laplacian=True, k=N_NEIGHBORS, n_jobs=N_JOBS)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Handwritten digits data:', fontsize=24)
ax1.scatter(sk_se[:, 0], sk_se[:, 1], c=digits.target, cmap='Spectral', s=5)
ax1.set_title('Sklearn\'s Laplacian Eigenmaps', fontsize=20)
ax2.scatter(flapmap_diff[:, 0], flapmap_diff[:, 1], c=digits.target, cmap='Spectral', s=5)
ax2.set_title('Fast Laplacian Eigenmaps with diffusion harmonics', fontsize=20)
ax3.scatter(flapmap_fuzzy[:, 0], flapmap_fuzzy[:, 1], c=digits.target, cmap='Spectral', s=5)
ax3.set_title('Fast Laplacian Eigenmaps with fuzzy simplicial sets', fontsize=20)
plt.show()
```
![](figs/Embedding_comparison.png)

As we can see, results are nearly identical.

# Benchmark

See the runtime comparison between this implementation and scikit-learn:

```
## Load benchmark function:
from fastlapmap.benchmark import runtime_benchmark

# Load data
from sklearn.datasets import load_digits
digits = load_digits()
data = digits.data

# Define hyperparameters
N_EIGS = 2
N_NEIGHBORS = 10
N_JOBS = 10
SIZES = [1000, 5000, 10000, 25000, 50000, 100000]
N_RUNS = 3

runtime_benchmark(data,
                  n_eigs=N_EIGS,
                  n_neighbors=N_NEIGHBORS,
                  n_jobs=N_JOBS,
                  sizes=SIZES,
                  n_runs=N_RUNS)
```

![](figs/Runtime_benchmark.png)

As you can see, the diffusion harmoics model is the fastest, followed closely by fuzzy simplicial sets. Both outperform 
scikit-learn default implementation and escalate linearly with sample size.