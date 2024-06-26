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
    jac_batched = jacobian(G(gs), x) # shape (fns, batch_size, batch_size, dims)
    r = jac_batched.permute(1, 3, 0, 2).diagonal(dim1=-2, dim2=-1).permute(2, 0, 1)
    return r



def rattle_step(x, v1, h, M, gs, e):
    '''
    Defining a function to take a step in the position, velocity form.
    g should be a vector-valued function of constraints.
    :return: x_1, v_1
    '''

    # M1 =  torch.inverse(M) commenting this out since we use the identity

    G1 = G(gs)


    DV = torch.zeros_like(x)
    batch_size = x.shape[0]
    DV_col = DV.reshape(batch_size,-1, 1)

    
    x_col = x.reshape(batch_size,-1, 1)
    v1_col = v1.reshape(batch_size,-1, 1)

    # doing Newton-Raphson iterations
    x2 = x_col + h * v1_col - 0.5*(h**2)* torch.bmm(M, DV_col)
    Q_col = x2
    Q = torch.squeeze(Q_col)
    J1 = J(gs, torch.squeeze(x_col))

    for _ in range(3):
        J2 = J(gs, torch.squeeze(Q))
        R = torch.bmm(torch.bmm(J2,M),J1.mT)
        dL = torch.bmm(torch.linalg.inv(R),G1(Q).unsqueeze(-1))
        Q= Q- torch.bmm(torch.bmm(M,J1.mT), dL).squeeze(-1)

    # half step for velocity
    Q_col = Q.reshape(batch_size,-1,1)
    v1_half = (Q_col - x_col)/h
    x_col = Q_col
    J1 = J(gs, torch.squeeze(x_col))

    # getting the level
    J2 = J(gs, torch.squeeze(Q))
    P = torch.bmm(torch.bmm(J1, M),J1.mT)
    T = torch.bmm(J1, (2/h * v1_half - torch.bmm(M, DV_col)))

    #solving the linear system
    L = torch.linalg.solve(P,T)

    v1_col = v1_half - h/2 * DV_col - h/2 * torch.bmm(J2.mT,L)

    return torch.squeeze(x_col), torch.squeeze(v1_col)


def gBAOAB_step(q_init,p_init,F, gs, h,M, gamma, k, kr,e):
    # setting up variables
    M1 = M
    batch_size = q_init.shape[0]
    R = torch.randn(batch_size, q_init.shape[1])
    p = p_init
    q = q_init

    a2 = torch.exp(torch.tensor(-gamma*h))
    b2 = torch.sqrt(k*(1-a2**(2)))


    # doing the initial p-update
    J1 = J(gs,q)
    G = J1
    raise ValueError(G.shape)
    to_invert = torch.bmm(torch.bmm(G, M1), torch.transpose(G,-2,-1))
    t2 = torch.bmm(torch.inverse(to_invert), torch.bmm(G , M1))
    # raise ValueError(torch.bmm(torch.transpose(G,-2,-1),t2).shape)
    L1 = torch.eye(q_init.shape[1]) - torch.bmm(torch.transpose(G,-2,-1),t2)
    p = p-  h/2 * torch.bmm(L1, F(q).unsqueeze(-1)).squeeze(-1)


    # doing the first RATTLE step
    for _ in range(kr):
      q, p = rattle_step(q, p, h/2*kr, M, gs, e)


    # the second p-update - (O-step in BAOAB)
    J2 = J(gs,q)
    G = J2
    to_invert=torch.bmm(G, torch.bmm(M1,torch.transpose(G,-1,-2)))
    # raise ValueError(torch.bmm(G, M1).shape, torch.bmm(torch.transpose(G,-2,-1),torch.inverse(to_invert)).shape)
    L2 = torch.eye(q_init.shape[1]) - torch.bmm(torch.bmm(torch.transpose(G,-2,-1),torch.inverse(to_invert)), torch.bmm(G, M1))
    p = a2* p + b2* torch.bmm(torch.bmm(torch.bmm(M**(1/2),L2), M**(1/2)), R.unsqueeze(-1)).squeeze(-1)

    # doing the second RATTLE step
    for i in range(kr):
      q, p = rattle_step(q, p, h/2*kr, M, gs, e)


    # the final p update
    J3= J(gs,q)
    G = J3

    L3 = torch.eye(q_init.shape[1]) - torch.bmm(torch.bmm(torch.bmm(torch.transpose(G,-2,-1), torch.inverse(G@ M1@ torch.transpose(G,-2,-1))), G), M1)
    p = p-  h/2 * torch.bmm(L3, F(q).unsqueeze(-1)).squeeze(-1)

    return q,p


def gBAOAB_integrator(q_init,p_init,F, gs, h,M, gamma, k, steps,kr,e):
    positions = []
    velocities = []
    q = q_init
    p = p_init
    for _ in range(steps):
        q, p = gBAOAB_step(q,p, F,gs, h,M, gamma, k,kr,e)
        positions.append(q)
        velocities.append(p)

    return positions, velocities


def cotangent_projection(gs):
    def proj(x):
        G = J(gs,x)
        M = torch.eye(G.size()[2]).broadcast_to(x.shape[0],G.size()[2],G.size()[2])
        # print(G.shape, M.shape)
        # print((G.mT).shape,torch.bmm(G,M).shape)
        to_invert = torch.bmm(torch.bmm(G,M),G.mT)

        L= torch.eye(G.size()[2]) - torch.bmm(torch.bmm(G.mT, torch.inverse(to_invert)) ,torch.bmm(G ,torch.inverse(M)))
        return L, G
    return proj