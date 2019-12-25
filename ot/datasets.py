"""
Simple example datasets for OT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import torch
import scipy as sp
from .utils import check_random_state, deprecated


def make_1D_gauss(n, m, s):
    """return a 1D histogram for a gaussian distribution (n bins, mean m and std s)

    Parameters
    ----------
    n : int
        number of bins in the histogram
    m : float
        mean value of the gaussian distribution
    s : float
        standard deviaton of the gaussian distribution

    Returns
    -------
    h : ndarray (n,)
        1D histogram for a gaussian distribution
    """
    x = torch.arange(n, dtype=torch.float64)
    h = torch.exp(-(x - m)**2 / (2 * s**2))
    return h / h.sum()


@deprecated()
def get_1D_gauss(n, m, sigma):
    """ Deprecated see  make_1D_gauss   """
    return make_1D_gauss(n, m, sigma)


def make_2D_samples_gauss(n, m, sigma, random_state=None):
    """Return n samples drawn from 2D gaussian N(m,sigma)

    Parameters
    ----------
    n : int
        number of samples to make
    m : ndarray, shape (2,)
        mean value of the gaussian distribution
    sigma : ndarray, shape (2, 2)
        covariance matrix of the gaussian distribution
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : ndarray, shape (n, 2)
        n samples drawn from N(m, sigma).
    """

    generator = check_random_state(random_state)
    if torch.isscalar(sigma):
        sigma = torch.array([sigma, ])
    if len(sigma) > 1:
        P = sp.linalg.sqrtm(sigma)
        res = generator.randn(n, 2).matmul(P) + m
    else:
        res = generator.randn(n, 2) * torch.sqrt(sigma) + m
    return res


@deprecated()
def get_2D_samples_gauss(n, m, sigma, random_state=None):
    """ Deprecated see  make_2D_samples_gauss   """
    return make_2D_samples_gauss(n, m, sigma, random_state=None)


def make_data_classif(dataset, n, nz=.5, theta=0, random_state=None, **kwargs):
    """Dataset generation for classification problems

    Parameters
    ----------
    dataset : str
        type of classification problem (see code)
    n : int
        number of training samples
    nz : float
        noise level (>0)
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : ndarray, shape (n, d)
        n observation of size d
    y : ndarray, shape (n,)
        labels of the samples.
    """
    generator = check_random_state(random_state)

    if dataset.lower() == '3gauss':
        y = torch.floor((np.arange(n) * 1.0 / n * 3)) + 1
        x = torch.zeros((n, 2))
        # class 1
        x[y == 1, 0] = -1.
        x[y == 1, 1] = -1.
        x[y == 2, 0] = -1.
        x[y == 2, 1] = 1.
        x[y == 3, 0] = 1.
        x[y == 3, 1] = 0

        x[y != 3, :] += 1.5 * nz * generator.randn(sum(y != 3), 2)
        x[y == 3, :] += 2 * nz * generator.randn(sum(y == 3), 2)

    elif dataset.lower() == '3gauss2':
        y = torch.floor((np.arange(n) * 1.0 / n * 3)) + 1
        x = torch.zeros((n, 2))
        y[y == 4] = 3
        # class 1
        x[y == 1, 0] = -2.
        x[y == 1, 1] = -2.
        x[y == 2, 0] = -2.
        x[y == 2, 1] = 2.
        x[y == 3, 0] = 2.
        x[y == 3, 1] = 0

        x[y != 3, :] += nz * generator.randn(sum(y != 3), 2)
        x[y == 3, :] += 2 * nz * generator.randn(sum(y == 3), 2)

    elif dataset.lower() == 'gaussrot':
        rot = torch.array(
            [[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]])
        m1 = torch.Tensor([-1, 1])
        m2 = torch.Tensor([1, -1])
        y = torch.floor((torch.arange(n) * 1.0 / n * 2)) + 1
        n1 = torch.sum(y == 1)
        n2 = torch.sum(y == 2)
        x = torch.zeros((n, 2))

        x[y == 1, :] = get_2D_samples_gauss(n1, m1, nz, random_state=generator)
        x[y == 2, :] = get_2D_samples_gauss(n2, m2, nz, random_state=generator)

        x = x.matmul(rot)

    else:
        x = torch.zeros(1)
        y = torch.zeros(1)
        print("unknown dataset")

    return x, y.int()


@deprecated()
def get_data_classif(dataset, n, nz=.5, theta=0, random_state=None, **kwargs):
    """ Deprecated see  make_data_classif   """
    return make_data_classif(dataset, n, nz=.5, theta=0, random_state=None, **kwargs)
