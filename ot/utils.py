import multiprocessing
from functools import reduce
import time

import torch
import torch.nn.functional as F
import numpy as np
import sys
import warnings
from scipy.spatial.distance import cdist

try:
    from inspect import signature
except ImportError:
    from .externals.funcsigs import signature

__time_tic_toc = time.time()

def unif(n):

    """return a uniform histogram of length n (simplex)
    
    Parameters
    ----------
    n : int
        number of bins in the histogram
    Returns
    -------
    h : torch.Tensor (n,)
        histogram of length n such that h_i=1/n for all i
    """
    return torch.ones(n)/n

def clean_zeros(a, b, M):
    """ Remove all components with zeros weights in a and b
    """
    M2 = M[a > 0, :][:, b > 0].copy()  # copy force c style matrix (froemd)
    a2 = a[a > 0]
    b2 = b[b > 0]
    return a2, b2, M2

def euclidean_distances(X, Y, squared=False):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.
    Parameters
    ----------
    X : {tensor-like}, shape (n_samples_1, n_features)
    Y : {tensor-like}, shape (n_samples_2, n_features)
    squared : boolean, optional
        Return squared Euclidean distances.
    Returns
    -------
    distances : {Tensor}, shape (n_samples_1, n_samples_2)
    """

    

    X_col = X.unsqueeze(1)
    Y_lin = Y.unsqueeze(0)
 
    if squared == True:
        c = torch.sum((torch.abs(X_col - Y_lin)) ** 2, 2).sqrt_()
    else:
        c = torch.sum((torch.abs(X_col - Y_lin)) ** 2, 2)
    

    return c

def dist(x1, x2=None, metric='sqeuclidean'):

    """Compute distance between samples in x1 and x2 using function scipy.spatial.distance.cdist
    Parameters
    ----------
    x1 : ndarray, shape (n1,d)
        matrix with n1 samples of size d
    x2 : array, shape (n2,d), optional
        matrix with n2 samples of size d (if None then x2=x1)
    metric : str | callable, optional
        Name of the metric to be computed (full list in the doc of scipy),  If a string,
        the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns
    -------
    M : np.array (n1,n2)
        distance matrix computed with given metric
    """

    if x2 is None:
        x2 = x1
    if metric == "sqeuclidean":
        return euclidean_distances(x1, x2, squared=True)
    return cdist(x1, x2, metric=metric)

def dist0(n, method='lin_square'):
    """Compute standard cost matrices of size (n, n) for OT problems
    Parameters
    ----------
    n : int
        Size of the cost matrix.
    method : str, optional
        Type of loss matrix chosen from:
        * 'lin_square' : linear sampling between 0 and n-1, quadratic loss
    Returns
    -------
    M : ndarray, shape (n1,n2)
        Distance matrix computed with given metric.
    """
    res = 0
    if method == 'lin_square':

        x = torch.arange(0, n, 1, dtype=torch.float).reshape((n, 1))
        # x = np.arange(n, dtype=np.float64).reshape((n, 1))
        res = dist(x, x)
    return res

def cost_normalization(C, norm=None):
    """ Apply normalization to the loss matrix
    Parameters
    ----------
    C : ndarray, shape (n1, n2)
        The cost matrix to normalize.
    norm : str
        Type of normalization from 'median', 'max', 'log', 'loglog'. Any
        other value do not normalize.
    Returns
    -------
    C : ndarray, shape (n1, n2)
        The input cost matrix normalized according to given norm.
    """

    if norm is None:
        pass
    elif norm == "median":
        C /= float(torch.median(C))
    elif norm == "max":
        C /= float(torch.max(C)[0])
    elif norm == "log":
        C = torch.log(1 + C)
    elif norm == "loglog":
        C = torch.log1p(torch.log1p(C))
    else:
        raise ValueError('Norm %s is not a valid option.\n'
                         'Valid options are:\n'
                         'median, max, log, loglog' % norm)
    return C

def dots(*args):
    """ dots function for multiple matrix multiply """
    return reduce(torch.dot, args)


def fun(f, q_in, q_out):
    """ Utility function for parmap with no serializing problems """
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))




