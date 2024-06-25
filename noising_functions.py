import torch
import torch.nn as nn
import numpy as np
from torch.autograd.functional import jacobian
import tqdm


def G(gs):
    '''
    each g in gs should act on batches, eg lambda x: (x[:,0] -  x[:,1])**2
    :param gs: a list of tensor functions
    :return: a function sending a tensor to the stacked matrix of the functions of that tensor
    '''
    def G_gs(tensor):
        x = torch.squeeze(tensor)
        return torch.stack([g(x) for g in gs], 1)
    return G_gs


def J(gs, x):
    '''Returns the Jacobian evaluated at x for a list gs of constraint functions'''
    return jacobian(G(gs), torch.squeeze(x))

def rattle_step(x, v1, h, M, gs, e):
    '''
    Defining a function to take a step in the position, velocity form.
    g should be a vector-valued function of constraints.
    :return: x_1, v_1
    '''

    # M1 =  torch.inverse(M) commenting this out since we use the identity

    G1 = G(gs)


    DV = torch.zeros_like(x)

    DV_col = DV.reshape(-1, 1)

    batch_size = x.shape[0]
    x_col = x.reshape(batch_size,-1, 1)
    v1_col = v1.reshape(batch_size,-1, 1)

    # doing Newton-Raphson iterations
    x2 = x_col + h * v1_col - 0.5*(h**2)* torch.bmm(M, DV_col)
    Q_col = x2
    Q = torch.squeeze(Q_col)
    J1 = J(gs, torch.squeeze(x_col))

    for _ in range(3):
        J2 = J(gs, torch.squeeze(Q))
        R = torch.bmm(torch.bmm(J2,M),J1.t())
        dL = torch.bmm(torch.linalg.inv(R),G1(Q))
        Q= Q- torch.bmm(torch.bmm(M,J1.t()), dL)

    # half step for velocity
    Q_col = Q.reshape(batch_size,-1,1)
    v1_half = (Q_col - x_col)/h
    x_col = Q_col
    J1 = J(gs, torch.squeeze(x_col))

    # getting the level
    J2 = J(gs, torch.squeeze(Q))
    P = torch.bmm(torch.bmm(J1, M),J1.t())
    T = torch.bmm(J1, (2/h * v1_half - torch.bmm(M, DV_col)))

    #solving the linear system
    L = torch.linalg.solve(P,T)

    v1_col = v1_half - h/2 * DV_col - h/2 * torch.bmm(J2.t(),L)

    return torch.squeeze(x_col), torch.squeeze(v1_col)