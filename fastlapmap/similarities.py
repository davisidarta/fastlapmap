import numpy as np
from fastlapmap.ann import NMSlibTransformer
from scipy.sparse import find, coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors

from . import ann


SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def fuzzy_simplicial_set_ann(
        X,
        n_neighbors=15,
        knn_indices=None,
        knn_dists=None,
        backend='nmslib',
        metric='cosine',
        n_jobs=None,
        efC=50,
        efS=50,
        M=15,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        apply_set_operations=True,
        return_dists=False,
        verbose=False):
    """
    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union. This algorithm
    was first implemented and made popular in [UMAP](https://arxiv.org/abs/1802.03426).

    ----------
    Parameters
    ----------
    `X` : array of shape (n_samples, n_features).
        The data to be modelled as a fuzzy simplicial set.

    `n_neighbors` : int.
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.

    `backend` : str (optional, default 'nmslib').
        Which backend to use to compute nearest-neighbors. Options for fast, approximate nearest-neighbors
        are 'nmslib' (default) and 'hnswlib'. For exact nearest-neighbors, use 'sklearn'.

    `metric` : str (optional, default 'cosine').
        Distance metric for building an approximate kNN graph. Defaults to
        'cosine'. Users are encouraged to explore different metrics, such as 'euclidean' and 'inner_product'.
        The 'hamming' and 'jaccard' distances are also available for string vectors.
         Accepted metrics include NMSLib*, HNSWlib** and sklearn metrics. Some examples are:

        -'sqeuclidean' (*, **)

        -'euclidean' (*, **)

        -'l1' (*)

        -'lp' - requires setting the parameter ``p`` (*) - similar to Minkowski

        -'cosine' (*, **)

        -'inner_product' (**)

        -'angular' (*)

        -'negdotprod' (*)

        -'levenshtein' (*)

        -'hamming' (*)

        -'jaccard' (*)

        -'jansen-shan' (*).

    `n_jobs` : int (optional, default 1).
        number of threads to be used in computation. Defaults to 1. The algorithm is highly
        scalable to multi-threading.

    `M` : int (optional, default 30).
        defines the maximum number of neighbors in the zero and above-zero layers during HSNW
        (Hierarchical Navigable Small World Graph). However, the actual default maximum number
        of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
        is 5-100. For more information on HSNW, please check https://arxiv.org/abs/1603.09320.
        HSNW is implemented in python via NMSlib. Please check more about NMSlib at https://github.com/nmslib/nmslib.

    `efC` : int (optional, default 100).
        A 'hnsw' parameter. Increasing this value improves the quality of a constructed graph
        and leads to higher accuracy of search. However this also leads to longer indexing times.
        A reasonable range for this parameter is 50-2000.

    `efS` : int (optional, default 100).
        A 'hnsw' parameter. Similarly to efC, increasing this value improves recall at the
        expense of longer retrieval time. A reasonable range for this parameter is 50-2000.

    `knn_indices` : array of shape (n_samples, n_neighbors) (optional).
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.

    `knn_dists` : array of shape (n_samples, n_neighbors) (optional).
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.

    `set_op_mix_ratio` : float (optional, default 1.0).
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    `local_connectivity` : int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    `verbose` : bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    `return_dists` : bool or None (optional, default none)
        Whether to return the pairwise distance associated with each edge.

    Returns
    -------
    fuzzy_simplicial_set : coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """
    if knn_indices is None or knn_dists is None:
        if verbose:
            print('Running fast approximate nearest neighbors with NMSLIB using HNSW...')
        if metric not in ['sqeuclidean',
                                 'euclidean',
                                 'l1',
                                 'cosine',
                                 'angular',
                                 'negdotprod',
                                 'levenshtein',
                                 'hamming',
                                 'jaccard',
                                 'jansen-shan']:
            print('Please input a metric compatible with NMSLIB when use_nmslib is set to True')
        knn_indices, knn_dists = approximate_n_neighbors(X,
                                                         n_neighbors=n_neighbors,
                                                         metric=metric,
                                                         backend=backend,
                                                         n_jobs=n_jobs,
                                                         efC=efC,
                                                         efS=efS,
                                                         M=M,
                                                         verbose=verbose)

    knn_dists = knn_dists.astype(np.float32)
    knn_dists = knn_dists

    sigmas, rhos = smooth_knn_dist(
        knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity),
    )

    rows, cols, vals = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos
    )

    result = coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        result = (
                set_op_mix_ratio * (result + transpose - prod_matrix)
                + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    result.eliminate_zeros()
    if return_dists:
        return result, sigmas, rhos, knn_dists
    else:
        return result, sigmas, rhos


def approximate_n_neighbors(data,
                            n_neighbors=15,
                            metric='cosine',
                            n_jobs=10,
                            efC=20,
                            efS=20,
                            M=10,
                            p=11/16,
                            dense=False,
                            verbose=False
                            ):
    """
    Simple function using NMSlibTransformer from topodata.ann. This implements a very fast
    and scalable approximate k-nearest-neighbors graph on spaces defined by nmslib.
    Read more about nmslib and its various available metrics at
    https://github.com/nmslib/nmslib.

    ----------
    Parameters
    ----------
    n_neighbors : number of nearest-neighbors to look for. In practice,
                     this should be considered the average neighborhood size and thus vary depending
                     on your number of features, samples and data intrinsic dimensionality. Reasonable values
                     range from 5 to 100. Smaller values tend to lead to increased graph structure
                     resolution, but users should beware that a too low value may render granulated and vaguely
                     defined neighborhoods that arise as an artifact of downsampling. Defaults to 15. Larger
                     values can slightly increase computational time.

    backend : str (optional, default 'hnwslib')
        Which backend to use to compute nearest-neighbors. Options for fast, approximate nearest-neighbors
        are 'hnwslib' (default) and 'nmslib'. For exact nearest-neighbors, use 'sklearn'.

    metric : str (optional, default 'cosine')
        Distance metric for building an approximate kNN graph. Defaults to
        'cosine'. Users are encouraged to explore different metrics, such as 'euclidean' and 'inner_product'.
        The 'hamming' and 'jaccard' distances are also available for string vectors.
         Accepted metrics include NMSLib*, HNSWlib** and sklearn metrics. Some examples are:

        -'sqeuclidean' (*, **)

        -'euclidean' (*, **)

        -'l1' (*)

        -'lp' - requires setting the parameter ``p`` (*) - similar to Minkowski

        -'cosine' (*, **)

        -'inner_product' (**)

        -'angular' (*)

        -'negdotprod' (*)

        -'levenshtein' (*)

        -'hamming' (*)

        -'jaccard' (*)

        -'jansen-shan' (*)

    p : int or float (optional, default 11/16 )
        P for the Lp metric, when ``metric='lp'``.  Can be fractional. The default 11/16 approximates
        an astroid norm with some computational efficiency (2^n bases are less painstakinly slow to compute).
        See https://en.wikipedia.org/wiki/Lp_space for some context.

    n_jobs : number of threads to be used in computation. Defaults to 10 (~5 cores).

    efC : increasing this value improves the quality of a constructed graph and leads to higher
             accuracy of search. However this also leads to longer indexing times. A reasonable
             range is 100-2000. Defaults to 100.

    efS : similarly to efC, improving this value improves recall at the expense of longer
             retrieval time. A reasonable range is 100-2000.

    M : defines the maximum number of neighbors in the zero and above-zero layers during HSNW
           (Hierarchical Navigable Small World Graph). However, the actual default maximum number
           of neighbors for the zero layer is 2*M. For more information on HSNW, please check
           https://arxiv.org/abs/1603.09320. HSNW is implemented in python via NMSLIB. Please check
           more about NMSLIB at https://github.com/nmslib/nmslib .

    `verbose` : bool (optional, default False).
        Whether to report information on the current progress of the algorithm.

    -------------
    Returns
    -------------
     k-nearest-neighbors indices and distances. Can be customized to also return
        return the k-nearest-neighbors graph and its gradient.

    Example
    -------------

    knn_indices, knn_dists = approximate_n_neighbors(data)


    """

    # Construct an approximate k-nearest-neighbors graph
    anbrs = NMSlibTransformer(n_neighbors=n_neighbors,
                                      metric=metric,
                                      p=p,
                                      method='hnsw',
                                      n_jobs=n_jobs,
                                      M=M,
                                      efC=efC,
                                      efS=efS,
                                      dense=dense,
                                      verbose=verbose).fit(data)
    knn_inds, knn_distances, grad, knn_graph = anbrs.ind_dist_grad(data)

    return knn_inds, knn_distances



def smooth_knn_dist(distances, k, n_iter=16, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.
    k: float
        The number of nearest neighbors to approximate for.
    n_iter: int (optional, default 16)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.
    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.
    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.
    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.
    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.
    rhos: array of shape(n_samples)
        The local connectivity adjustment.
    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)
    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)
    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


def diffusion_harmonics(X, n_neighbors=10, metric='euclidean', n_jobs=1, efC=20, efS=20, M=10, p=11/16, verbose=False):
    """

    Computes the [diffusion potential](https://doi.org/10.1073/pnas.0500334102) between samples using an anisotropic diffusion method
     (renormalized by median k-nearest-neighbor).

    ----------
    Parameters
    ----------
    `X` : input data. May be a numpy.ndarray, a pandas.DataFrame or a scipy.sparse.csr_matrix.numpy

    `n_neighbors` : int (optional, default 10).
        How many neighbors to use for computations.

    `metric` : str (optional, default 'euclidean').
        which metric to use when computing neighborhood distances. Defaults to 'euclidean'.
        Accepted metrics include:
        -'sqeuclidean'
        -'euclidean'
        -'l1'
        -'lp' - requires setting the parameter `p` - equivalent to minkowski distance
        -'cosine'
        -'angular'
        -'negdotprod'
        -'levenshtein'
        -'hamming'
        -'jaccard'
        -'jansen-shan'

    `p` : int or float (optional, default 11/16 )
        P for the Lp metric, when ``metric='lp'``.  Can be fractional. The default 11/16 approximates
        an astroid norm with some computational efficiency (2^n bases are less painstakinly slow to compute).
        See https://en.wikipedia.org/wiki/Lp_space for some context.

    `n_jobs` : int (optional, default 1).
        number of threads to be used in computation. Defaults to 1. The algorithm is highly
        scalable to multi-threading.

    `M` : int (optional, default 10).
        defines the maximum number of neighbors in the zero and above-zero layers during HSNW
        (Hierarchical Navigable Small World Graph). However, the actual default maximum number
        of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
        is 5-100. For more information on HSNW, please check https://arxiv.org/abs/1603.09320.
        HSNW is implemented in python via NMSlib. Please check more about NMSlib at https://github.com/nmslib/nmslib.

    `efC` : int (optional, default 20).
        A 'hnsw' parameter. Increasing this value improves the quality of a constructed graph
        and leads to higher accuracy of search. However this also leads to longer indexing times.
        A reasonable range for this parameter is 10-500.

    `efS` : int (optional, default 100).
        A 'hnsw' parameter. Similarly to efC, increasing this value improves recall at the
        expense of longer retrieval time. A reasonable range for this parameter is 10-500.

    `verbose` : bool (optional, default False).
        Whether to report information on the current progress of the algorithm.

    ----------
    Returns
    -------

    `W` : an affinity matrix encoding diffusion potential between samples.

    """
    N = np.shape(X)[0]
    knn = NMSlibTransformer(n_neighbors=n_neighbors,
                            metric=metric,
                            p=p,
                            n_jobs=n_jobs,
                            M=M,
                            efC=efC,
                            efS=efS,
                            verbose=verbose).fit_transform(X)
    median_k = np.floor(n_neighbors / 2).astype(np.int)
    adap_sd = np.zeros(np.shape(X)[0])
    for i in np.arange(len(adap_sd)):
        adap_sd[i] = np.sort(knn.data[knn.indptr[i]: knn.indptr[i + 1]])[
            median_k - 1
            ]
    x, y, dists = find(knn)
    dists = dists / (adap_sd[x] + 1e-10)
    W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])
    return W

def cknn_graph(X, n_neighbors=10, delta=1.0, metric='euclidean', n_jobs=1, efC=20, efS=20, M=10, p=11/16, t='inf',
                 include_self=True, is_sparse=False, return_adj=True, verbose=False):
    """
    Continuous k-nearest-neighbors.  CkNN
    is the unique unweighted construction that yields a geometry consistent with
    the connected components of the underlying manifold in the limit of large
    data. See the [CkNN manuscript](http://dx.doi.org/10.3934/fods.2019001) for details.

    ----------
    Parameters
    ----------
    `X` : input data. May be a numpy.ndarray, a pandas.DataFrame or a scipy.sparse.csr_matrix.numpy

    `n_neighbors` : int (optional, default 5)
            Number of neighbors to estimate the density around the point.
            It appeared as a parameter `k` in the paper.

    `delta` : float (optional, default 1.0)
            A parameter to decide the radius for each points. The combination
            radius increases in proportion to this parameter
            .
    `metric` : str (optional, default 'euclidean').
        which metric to use when computing neighborhood distances. Defaults to 'euclidean'.
        Accepted metrics include:
        -'sqeuclidean'
        -'euclidean'
        -'l1'
        -'lp' - requires setting the parameter `p` - equivalent to minkowski distance
        -'cosine'
        -'angular'
        -'negdotprod'
        -'levenshtein'
        -'hamming'
        -'jaccard'
        -'jansen-shan'

    `n_jobs` : int (optional, default 1).
        number of threads to be used in computation. Defaults to 1. The algorithm is highly
        scalable to multi-threading.

    `M` : int (optional, default 10).
        defines the maximum number of neighbors in the zero and above-zero layers during HSNW
        (Hierarchical Navigable Small World Graph). However, the actual default maximum number
        of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
        is 5-100. For more information on HSNW, please check https://arxiv.org/abs/1603.09320.
        HSNW is implemented in python via NMSlib. Please check more about NMSlib at https://github.com/nmslib/nmslib.

    `efC` : int (optional, default 20).
        A 'hnsw' parameter. Increasing this value improves the quality of a constructed graph
        and leads to higher accuracy of search. However this also leads to longer indexing times.
        A reasonable range for this parameter is 10-500.

    `efS` : int (optional, default 100).
        A 'hnsw' parameter. Similarly to efC, increasing this value improves recall at the
        expense of longer retrieval time. A reasonable range for this parameter is 10-500.

    `t` : 'inf' or float or int (optional, default='inf')
            The decay parameter of heat kernel. The weights are calculated as
            follow:
                W_{ij} = exp(-(||x_{i}-x_{j}||^2)/t)
            For more infomation, read the paper 'Laplacian Eigenmaps for
            Dimensionality Reduction and Data Representation', Belkin, et. al.

    `include_self` : bool (optional, default True).
            All diagonal elements are 1.0 if this parameter is True.

    `is_sparse` : bool (optional, default True).
            Returns a scipy.sparse.csr_matrix object if this
            parameter is True. Otherwise, returns numpy.ndarray object.

    `return_adj` : bool (optional, default False)
            Whether to return the adjacency matrix instead.

    `verbose` : bool (optional, default False).
        Whether to report information on the current progress of the algorithm.
    ----------
    Returns
    ----------

    The affinity (or adjacency) matrix as a scipy.sparse.csr_matrix or numpy.ndarray object, depending on `is_sparse`
        and on `return_adj`.

    """


    n_samples = X.shape[0]

    if n_neighbors < 1 or n_neighbors > n_samples-1:
        raise ValueError("`n_neighbors` must be "
                         "in the range 1 to number of samples")
    if len(X.shape) != 2:
        raise ValueError("`X` must be 2D matrix")
    if n_samples < 2:
        raise ValueError("At least 2 data points are required")

    if metric == 'precomputed':
        if X.shape[0] != X.shape[1]:
            raise ValueError("`X` must be square matrix")
        dmatrix = X
    else:
        knn = NMSlibTransformer(n_neighbors=n_neighbors,
                                metric=metric,
                                p=p,
                                n_jobs=n_jobs,
                                M=M,
                                efC=efC,
                                efS=efS,
                                verbose=verbose).fit_transform(X)
        dmatrix = knn.toarray()
    darray_n_nbrs = np.partition(dmatrix, n_neighbors)[:, [n_neighbors]]
    # prevent approximately null results (div by 0)
    div_matrix = np.sqrt(darray_n_nbrs.dot(darray_n_nbrs.T)) + 1e-12
    ratio_matrix = dmatrix / div_matrix
    diag_ptr = np.arange(n_samples)

    if isinstance(delta, (int, float)):
        ValueError("Invalid argument type. "
                   "Type of `delta` must be float or int")
    A = csr_matrix(ratio_matrix < delta)

    if include_self:
        A[diag_ptr, diag_ptr] = True
    else:
        A[diag_ptr, diag_ptr] = False

    if return_adj:
        return A
    else:
        if t == 'inf':
            K = A.astype(np.float)
        else:
            mask = A.nonzero()
            weights = np.exp(-np.power(dmatrix[mask], 2)/t)
            dmatrix[:] = 0.
            dmatrix[mask] = weights
            K = csr_matrix(dmatrix)
        if not is_sparse:
            K = K.toarray()

        return K

