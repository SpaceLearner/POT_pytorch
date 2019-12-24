import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

# from .utils import unif, dist

def sinkhorn(a, b, M, reg, method='sinkhorn', numItermax=1000):

    # if method.lower() == 'sinkhorn':

    pass

def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False, **kwargs):

    a = torch.Tensor(a)
    b = torch.Tensor(b)
    M = torch.Tensor(M)

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
            








    
