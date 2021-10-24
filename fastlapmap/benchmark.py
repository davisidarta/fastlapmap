## Load some libraries:
import time
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.manifold import SpectralEmbedding
from fastlapmap.spectral import LapEigenmap
from sklearn.utils import resample
from pylab import *
import matplotlib.pyplot as plt
import seaborn as sns


def runtime_benchmark(data, n_eigs=2, n_neighbors=10, n_jobs=10, sizes=[1000, 5000, 10000], n_runs=3):
    algorithms = [
        'fastlapmap with diffusion harmonics',
        'fastlapmap with fuzzy simplicial sets',
        'scikit-learn'
    ]
    performance_data = {}

    for algorithm in algorithms:
        alg_name = str(algorithm).split('(')[0]
        performance_data[alg_name] = data_size_scaling(algorithm, data, n_eigs=n_eigs,
                                                       n_neighbors=n_neighbors, n_jobs=n_jobs,
                                                       sizes=sizes, n_runs=n_runs)
        print(f"[{time.asctime(time.localtime())}] Completed {alg_name}")

    for alg_name, perf_data in performance_data.items():
        sns.regplot('dataset size', 'runtime (s)', perf_data, order=2, label=alg_name)

    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(params)
    plt.legend(fontsize=20)
    plt.xlabel('Sample size', fontsize=20)
    plt.ylabel('Runtime (s)', fontsize=20)
    plt.show()

def data_size_scaling(algorithm, data, n_eigs=2, n_neighbors=10, n_jobs=10, sizes=[1000, 5000, 10000], n_runs=3):
    result = []
    for run in range(n_runs):
        for size in sizes:
            subsample = resample(data, n_samples=size)
            subsample_sparse = csr_matrix(subsample)
            start_time = time.time()
            if algorithm == 'fastlapmap with diffusion harmonics':
                # Use sparse matrix for speed
                fast_lapmap = LapEigenmap(subsample_sparse, n_eigs=n_eigs, k=n_neighbors, similarity='diffusion', n_jobs=n_jobs)
            elif algorithm == 'fastlapmap with fuzzy simplicial sets':
                # Use sparse matrix for speed
                fast_lapmap = LapEigenmap(subsample_sparse, n_eigs=n_eigs, k=n_neighbors, similarity='fuzzy', n_jobs=n_jobs)
            elif algorithm == 'scikit-learn':
                # sklearn cannot use nearest_neighbors to build affinities with sparse matrices
                # when inputed a sparse matrix, it uses a radial basis function, which is *significantly* more expensive
                sklearn_lapmap = SpectralEmbedding(n_components=n_eigs, n_neighbors=n_neighbors, n_jobs=n_jobs).fit_transform(subsample)
            elapsed_time = time.time() - start_time
            result.append((size, elapsed_time))
    return pd.DataFrame(result, columns=('dataset size', 'runtime (s)'))

