import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

# from .utils import unif, dist

def sinkhorn(a, b, M, reg, method='sinkhorn', numItermax=1000):

    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix
    """

    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp(a, b, M, numItermax=numItermax,stopThr=stopThr, verbose=verbose, log=log, **kwargs)
    elif method.lower() == 'greenkhorn':
        return greenkhorn(a, b, M, reg, numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log)
    elif method.lower() == 'sinkhorn_stabilized':
        return sinkhorn_stabilized(a, b, M, reg, numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log, **kwargs)
    elif method.lower() == 'sinkhorn_epsilon_scaling':
        return sinkhorn_epsilon_scaling(a, b, M, reg,numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log, **kwargs)
    else: 
        raise ValueError("Unknown method '%s'."% method)


def sinkhorn2(a, b, M, reg, method='sinkhorn', numItermax=1000,stopThr=1e-9, verbose=False, log=False, **kwargs):

    r"""
    Solve the entropic regularization optimal transport problem and return the loss
    """
    b = torch.Tensor(b).float()
    if len(b.shape) < 2:
        b = b[:, None]
    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp(a, b, M, numItermax=numItermax,stopThr=stopThr, verbose=verbose, log=log, **kwargs)
    elif method.lower() == 'sinkhorn_stabilized':
        return sinkhorn_stabilized(a, b, M, reg, numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log, **kwargs)
    elif method.lower() == 'sinkhorn_epsilon_scaling':
        return sinkhorn_epsilon_scaling(a, b, M, reg,numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log, **kwargs)
    else: 
        raise ValueError("Unknown method '%s'."% method)


def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False, **kwargs):

    a = torch.Tensor(a).float()
    b = torch.Tensor(b).float()
    M = torch.Tensor(M).float()

    if len(a) == 0:
        a = torch.ones(M.shape[0], dtype=torch.float64) / M.shape[0]
    if len(b) == 0:
        b = torch.ones(M.shape[1], dtype=torch.float64) / M.shape[1]

    # init data
    dim_a = len(a)
    dim_b = len(b)

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = torch.ones(dim_a, n_hists) / dim_a
        v = torch.ones(dim_b, n_hists) / dim_b
    else:
        u = torch.ones(dim_a) / dim_a
        v = torch.ones(dim_b) / dim_b

    # K = torch.empty(M.shape, dtype=M.dtype)
    # np.divide(M, -reg, out=K)
    # np.exp(K, out=K)

    K = torch.exp(M / -reg)

    tmp2 = torch.empty(b.shape, dtype=M.dtype)

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1

    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        
        KtransposeU = torch.matmul(K.T, u)
        v = b.div(KtransposeU + 1e-32)
        u = 1. / torch.matmul(Kp, v)

        if (KtransposeU == 0).any() \
            or torch.isnan(u).any() or torch.isnan(v).any() \
            or torch.isinf(u).any() or torch.isinf(u).any():
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                tmp2 = torch.einsum('ik,ij,jk->jk', u, K, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = torch.einsum('i,ij,j->j', u, K, v)
            err = torch.norm(tmp2 - b)

            if log:
                log['err'].append(err.item())
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
    if log:
        log['u'] = u
        log['v'] = v

    if n_hists:  # return only loss
        res = torch.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res
    
    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


def sinkhorn_stabilized(a, b, M, reg, numItermax=1000, tau=1e3, stopThr=1e-9, warmstart=None, verbose=False, print_period=20, log=False, **kwargs):

    r"""
    Solve the entropic regularization OT problem with log stabilization 
    """

    a = torch.Tensor(a).float()
    b = torch.Tensor(b).float()
    M = torch.Tensor(M).float()

    if len(a) == 0:
        a = torch.ones(M.shape[0], dtype=torch.float64) / M.shape[0]
    if len(b) == 0:
        b = torch.ones(M.shape[1], dtype=torch.float64) / M.shape[1]

    # test if multiple target
    if len(b.shape) > 1:
        n_hists = b.shape[1]
        a.unsqueeze_(-1)
    else:
        n_hists = 0

    # init data
    dim_a = len(a)
    dim_b = len(b)

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = torch.zeros(dim_a), torch.zeros(dim_b)
    else:
        alpha, beta = warmstart

    if n_hists:
        u = torch.ones(dim_a, n_hists) / dim_a
        v = torch.ones(dim_b, n_hists) / dim_b
    else:
        u, v = torch.ones(dim_a) / dim_a, torch.ones(dim_b) / dim_b

    def get_K(alpha, beta):
        """log space computation"""
        return torch.exp(-(M - alpha.reshape((dim_a, 1))
                        - beta.reshape((1, dim_b))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return torch.exp(-(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b)))
                      / reg + torch.log(u.reshape((dim_a, 1))) + torch.log(v.reshape((1, dim_b))))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1

    while loop:
    
        uprev = u
        vprev = v

        # sinkhorn update
        v = b / (torch.matmul(K.T, u) + 1e-16)
        u = a / (torch.matmul(K, v) + 1e-16)

        # remove numerical problems and store them in K
        if torch.abs(u).max() > tau or torch.abs(v).max() > tau:
            if n_hists:
                alpha, beta = alpha + reg * \
                    torch.max(torch.log(u), 1)[0], beta + reg * torch.max(torch.log(v))[0]
            else:
                alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)
                if n_hists:
                    u, v = torch.ones((dim_a, n_hists)) / dim_a, torch.ones((dim_b, n_hists)) / dim_b
                else:
                    u, v = torch.ones(dim_a) / dim_a, torch.ones(dim_b) / dim_b
            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                err_u = abs(u - uprev).max()
                err_u /= max(abs(u).max(), abs(uprev).max(), 1.)
                err_v = abs(v - vprev).max()
                err_v /= max(abs(v).max(), abs(vprev).max(), 1.)
                err = 0.5 * (err_u + err_v)
            else:
                transp = get_Gamma(alpha, beta, u, v)
                err = torch.norm((torch.sum(transp, 0) - b))
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % (print_period * 20) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        if torch.isnan(u).any() or torch.isnan(v).any():
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        cpt = cpt + 1

    if log:
        if n_hists:
            alpha = alpha[:, None]
            beta = beta[:, None]
        logu = alpha / reg + torch.log(u)
        logv = beta / reg + torch.log(v)
        log['logu'] = logu
        log['logv'] = logv
        log['alpha'] = alpha + reg * torch.log(u)
        log['beta'] = beta + reg * torch.log(v)
        log['warmstart'] = (log['alpha'], log['beta'])
        if n_hists:
            res = torch.zeros((n_hists))
            for i in range(n_hists):
                res[i] = torch.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
            return res, log

        else:
            return get_Gamma(alpha, beta, u, v), log
    else:
        if n_hists:
            res = torch.zeros((n_hists))
            for i in range(n_hists):
                res[i] = torch.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
            return res
        else:
            return get_Gamma(alpha, beta, u, v)


def greenkhorn(a, b, M, reg, numItermax=10000, stopThr=1e-9, verbose=False, log=log):

    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix
    """

    a = torch.Tensor(a).float()
    b = torch.Tensor(b).float()
    M = torch.Tensor(M).float()

    if len(a) == 0:
        a = torch.ones(M.shape[0], dtype=torch.float64) / M.shape[0]
    if len(b) == 0:
        b = torch.ones(M.shape[1], dtype=torch.float64) / M.shape[1]

    dim_a = a.shape[0]
    dim_b = b.shape[0]

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = torch.empty_like(M)
    K = torch.exp(M / (-reg + 1e-32))

    u = torch.full(dim_a, 1. / dim_a)
    v = torch.full(dim_b, 1. / dim_b)
    u.unsqueeze_(-1)
    v.unsqueeze_(0)
    G = u * K * v

    viol = G.sum(1) - a
    viol_2 = G.sum(0) - b
    stopThr_val = 1

    if log:
        log = dict()
        log['u'] = u
        log['v'] = v

    for i in range(numItermax):
        i_1 = torch.argmax(torch.abs(viol))
        i_2 = torch.argmax(torch.abs(viol_2))
        m_viol_1 = torch.abs(viol[i_1])
        m_viol_2 = torch.abs(viol_2[i_2])
        stopThr_val = torch.max(m_viol_1, m_viol_2)

        if m_viol_1 > m_viol_2:
            old_u = u[i_1]
            u[i_1] = a[i_1] / (K[i_1, :].dot(v))
            G[i_1, :] = u[i_1] * K[i_1, :] * v

            viol[i_1] = u[i_1] * K[i_1, :].dot(v) - a[i_1]
            viol_2 += (K[i_1, :].T * (u[i_1] - old_u) * v)

        else:
            old_v = v[i_2]
            v[i_2] = b[i_2] / (K[:, i_2].T.dot(u))
            G[:, i_2] = u * K[:, i_2] * v[i_2]
            #aviol = (G@one_m - a)
            #aviol_2 = (G.T@one_n - b)
            viol += (-old_v + v[i_2]) * K[:, i_2] * u
            viol_2[i_2] = v[i_2] * K[:, i_2].dot(u) - b[i_2]

            #print('b',torch.max(abs(aviol -viol)),torch.max(abs(aviol_2 - viol_2)))

        if stopThr_val <= stopThr:
            break
    else:
        print('Warning: Algorithm did not converge')

    if log:
        log['u'] = u
        log['v'] = v

    if log:
        return G, log
    else:
        return G

    
def sinkhorn_epsilon_scaling(a, b, M, reg, numItermax=100, epsilon0=1e4, numInnerItermax=100, tau=1e3, stopThr=1e-9, warmstart=None, verbose=False, print_period=10, log=False, **kwargs):

    """
    Solve the entropic regularization optimal transport problem with log stabilization and epsilon scaling.
    """

    a = torch.Tensor(a).float()
    b = torch.Tensor(b).float()
    M = torch.Tensor(c).float()

    if len(a) == 0:
        a = torch.ones(M.shape[0], dtype=torch.float64) / M.shape[0]
    if len(b) == 0:
        b = torch.ones(M.shape[1], dtype=torch.float64) / M.shape[1]

    # init data
    dim_a = len(a)
    dim_b = len(b)

    numItermin = 35
    numItermax = max(numItermin, numItermax) 

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = torch.zeros(dim_a), torch.zeros(dim_b)
    else:
        alpha, beta = warmstart

    def get_K(alpha, beta):
        """log space computation"""
        return torch.exp(-(M - alpha.reshape((dim_a, 1))
                        - beta.reshape((1, dim_b))) / reg)

    # print(torch.min(K))
    def get_reg(n):  # exponential decreasing
        return (epsilon0 - reg) * torch.exp(-n) + reg

    loop = 1
    cpt = 0
    err = 1
    while loop:

        regi = get_reg(cpt)

        G, logi = sinkhorn_stabilized(a, b, M, regi,
                                      numItermax=numInnerItermax, stopThr=1e-9,
                                      warmstart=(alpha, beta), verbose=False,
                                      print_period=20, tau=tau, log=True)

        alpha = logi['alpha']
        beta = logi['beta']

        if cpt >= numItermax:
            loop = False

        if cpt % (print_period) == 0:  # spsion nearly converged
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = G
            err = torch.norm(
                (torch.sum(transp, axis=0) - b))**2 + torch.norm((torch.sum(transp, 1) - a))**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % (print_period * 10) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        if err <= stopThr and cpt > numItermin:
            loop = False

        cpt = cpt + 1
    # print('err=',err,' cpt=',cpt)
    if log:
        log['alpha'] = alpha
        log['beta'] = beta
        log['warmstart'] = (log['alpha'], log['beta'])
        return G, log
    else:
        return G


def geometricBar(weights, alldistribT):
    """return the weight geometric mean of distributions"""
    assert(len(weights) == alldistribT.shape[1])
    return torch.exp(torch.matmul(torch.log(alldistribT), weights.T))


def geometricMean(alldistribT):
    """return the  geometric mean of distributions"""
    return torch.exp(torch.mean(torch.log(alldistribT), dim=1))


def projR(gamma, p):
    """return the KL projection on the row constraints"""
    return torch.mul(gamma.T, p / torch.max(torch.sum(gamma, dim=1), torch.ones(1)*1e-10)).T


def projC(gamma, q):
    """return the KL projection on the column constrints """
    return np.mul(gamma, q / torch.max(np.sum(gamma, dim=0), torch.ones(1)*1e-10))


def barycenter(A, M, reg, weights=None, method="sinkhorn", numItermax=10000, stopThr=1e-4, verbose=False, log=False, **kwargs):

    r"""Compute the entropic regularized wasserstein barycenter of distributions A
    """

    if method.lower() == 'sinkhorn':
        return barycenter_sinkhorn(A, M, reg, weights=weights,
                                   numItermax=numItermax,
                                   stopThr=stopThr, verbose=verbose, log=log,
                                   **kwargs)
    elif method.lower() == 'sinkhorn_stabilized':
        return barycenter_stabilized(A, M, reg, weights=weights,
                                     numItermax=numItermax,
                                     stopThr=stopThr, verbose=verbose,
                                     log=log, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def barycenter_sinkhorn(A, M, reg, weights=None,numItermax=1000, stopThr=1e-4, verbose=False, log=False):

    r"""Compute the entropic regularized wasserstein barycenter of distributions A
    """

    if weights is None:
        weights = torch.ones(A.shape[1]).div(A.shape[1])
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    # M = M/np.median(M) # suggested by G. Peyre
    K = np.exp(-M / reg)

    cpt = 0
    err = 1

    UKv = torch.matmul(K, A.T.div(torch.sum(K, dim=0)).T)
    u = (geometricMean(UKv) / UKv.T).T

    while(err > stopThr and cpt < numItermax):
        cpt = cpt + 1
        UKv = u * torch.matmul(K, A.div(K.matmul(u)))
        u = (u.T * geometricBar(weights, UKv)).T.div(UKv)

        if cpt % 10 == 1:
            err = torch.sum(torch.std(UKv, dim=1))

            # log and verbose print
            if log:
                log['err'].append(err)
            
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
    
    if log:
        log['niter'] = cpt
        return geometricBar(weights, UKv), log
    else:
        return geometricBar(weights, UKv)

def barycenter_stabilized(A, M, reg, tau=1e10, weights=None,numItermax=1000, stopThr=1e-4, verbose=False, log=False):

    r"""Compute the entropic regularized wasserstein barycenter of distributions A with stabilization.
    """

    dim, n_hists = A.shape
    if weights is None:
        weights = torch.ones(n_hists).div(n_hists)
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}
    
    u = torch.ones(dim, n_hists).div(dim)
    v = torch.ones(dim, n_hists).div(dim)

    # print(reg)
    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = torch.empty(M.shape, dtype=M.dtype)
    torch.div(M, -reg, out=K)
    torch.exp(K, out=K)

    cpt = 0
    err = 1.
    alpha = torch.zeros(dim)
    beta = torch.zeros(dim)
    q = torch.ones(dim).div(dim)
    while(err > stopThr and cpt < numItermax):
        qprev = q
        Kv = K.matmul(v)
        u = A.div(Kv + 1e-32)
        Ktu = K.T.matmul(u)
        q = geometricBar(weights, Ktu)
        Q = q[:, None]
        v = Q.div(Ktu + 1e-32)
        absorbing = False
        if (u > tau).any() or (v > tau).any():
            absorbing = True
            alpha = alpha + reg * torch.log(torch.max(u, torch.ones(1)))
            beta = beta + reg * torch.log(torch.max(v, torch.ones(1)))
            K = torch.exp((alpha[:, None] + beta[None, :] - M).div(reg))
            v = torch.ones_like(v)
        Kv = K.matmul(v)
        if (Ktu == 0.).any() \
            or torch.isnan(u).any() or torch.isnan(v).any() \
            or torch.isinf(u).any() or torch.isinf(v).any()
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % cpt)
            q = qprev
            break
        if (cpt % 10 == 0 and not absorbing) or cpt ==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            err = abs(u * Kv - A).max()
            if log:
                log['err'].append(err)
             if verbose:
                    if cpt % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1
    if err > stopThr:
        warnings.warn("Stabilized Unbalanced Sinkhorn did not converge." + "Try a larger entropy `reg`" + "Or a larger absorption threshold `tau`.")
    if log:
        log['niter'] = cpt
        log['logu'] = torch.log(u + 1e-16)
        log['logv'] = torch.log(v + 1e-16)
        return q, log
    else:
        return q


def convolutional_barycenter2d(A, reg, weights=None, numItermax=10000, stopThr=1e-9, stabThr=1e-30, verbose=False, log=False):
    r"""Compute the entropic regularized wasserstein barycenter of distributions A where A is a collection of 2D images.
    """

     if weights is None:
        weights = torch.ones(A.shape[0]) / A.shape[0]
    else:
        assert(len(weights) == A.shape[0])

    if log:
        log = {'err': []}

    b = torch.zeros_like(A[0, :, :])
    U = torch.ones_like(A)
    KV = torch.ones_like(A)

    cpt = 0
    err = 1

    # build the convolution operator
    t = torch.linspace(0, 1, A.shape[1])
    Y, X = torch.meshgrid(t, t)
    xi1 = torch.exp(-(X - Y)**2 / reg)

    def K(x):
        return torch.matmul(torch.matmul(xi1, x), xi1)

    while (err > stopThr and cpt < numItermax):

        bold = b
        cpt = cpt + 1

        b = torch.zeros_like(A[0, :, :])
        for r in range(A.shape[0]):
            KV[r, :, :] = K(A[r, :, :] / torch.max(torch.ones(1)*stabThr, K(U[r, :, :])))
            b += weights[r] * torch.log(torch.max(torch.ones(1)*stabThr, U[r, :, :] * KV[r, :, :]))
        b = torch.exp(b)
        for r in range(A.shape[0]):
            U[r, :, :] = b / torch.max(torch.ones(1)*stabThr, KV[r, :, :])

        if cpt % 10 == 1:
            err = torch.sum(torch.abs(bold - b))
            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

    if log:
        log['niter'] = cpt
        log['U'] = U
        return b, log
    else:
        return b















    
