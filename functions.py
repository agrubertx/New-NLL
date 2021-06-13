import math
import numpy as np
import numpy.matlib as npm
import torch

from utils import R0, R0_grad  # for Ebola experiment

def test_func(x, example):
    '''
    A collection of some quantities that appeared in the original NLL paper
    and ours, plus some other miscellaneous ones.  This is the function to
    modify to use your own quantities.
    '''
    x = x.detach().cpu().numpy()

    size = x.shape
    Ns = size[0]
    dim = size[1]

    if example == 1:
        f = 0.5*(np.sin(2.0* math.pi * np.sum(x,axis=1)) + 1)
        df = npm.repmat(np.expand_dims(
                0.5 *2.0 * math.pi * np.cos(
                    2.0* math.pi * np.sum(x, axis=1)), axis=1), 1, dim)

    elif example == 2:
        x1 = np.copy(x)
        x1[:,0] = x1[:,0] - 0.5
        f = np.exp(-1.0 * np.sum(x1**2, axis=1))
        df = -2.0 * x1 * npm.repmat(np.expand_dims(f, axis=1),1,dim)

    elif example == 3:
        f = x[:,0]**3 + x[:,1]**3 + x[:,0] * 0.2 + 0.6 * x[:,1]
        df = npm.repmat(np.expand_dims(f,axis=1),1,dim)
        df[:,0] = 3.0*x[:,0]**2.0 + 0.2
        df[:,1] = 3.0*x[:,1]**2.0 + 0.6

    elif example == 4:
        f1 = 2.0*np.exp(-1 * np.sum((x-0.0) * (x-0.0), axis=1) * 2.0)
        f2 = 2.0*np.exp(-1 * np.sum((x-1.0) * (x-1.0), axis=1) * 2.0)
        f = f1 + f2
        df = -8.0 * (x-0.0) * npm.repmat(
                                np.expand_dims(f1, axis=1), 1, dim) -8.0 * (
                                    x-1.0) * npm.repmat(
                                        np.expand_dims(f2, axis=1), 1, dim)

    elif example == 5:
        cc = 0.1
        ww = 0.0
        f = np.sin(np.sum(x*cc,axis=1))
        df = npm.repmat(np.expand_dims(1 * cc * np.cos(
                                        np.sum(x,axis=1)), axis=1), 1, dim)

    elif example == 6:
        cc = 1.
        ww = 0.0
        f = np.prod((cc**(-2.0) + (x-ww)**2.0)**(-1.0), axis=1)
        df = npm.repmat(np.expand_dims(f,axis=1), 1, dim) * -1.0 * (
                cc**-2.0 + (x-ww)**2.0)**-1.0 * 2.0 * (x-ww)

    elif example == 7:
        cc = 0.1
        ww = 0.0
        f = (1.0 + np.sum(x*cc,axis=1))**-(dim+1)
        df = -(dim+1) * cc * npm.repmat(
                                np.expand_dims((1.0 + np.sum(
                                    x*cc,axis=1))**-(dim+2),axis=1),1, dim)

    elif example == 8:
        center = 0.0
        f = 0.5 * np.sum((x-center) * (x-center), axis=1)
        df = 1.0 * (x-center)

    elif example == 9:
        center = 0.0
        f = np.sin(np.sum((x-center) * (x-center), axis=1))
        ff  = np.cos(np.sum((x-center) * (x-center), axis=1))
        df = 2.0 * (x-center) * npm.repmat(np.expand_dims(ff, axis=1),1,dim)

    elif example == 10:
        center = 0.0
        f  = np.sin(np.sum((x-center)**3.0, axis=1))
        ff = np.cos(np.sum((x-center)**3.0, axis=1))
        df = 3.0 * (x-center)**2.0 * npm.repmat(
                                        np.expand_dims(ff, axis=1),1,dim)

    elif example == 11:   # requires dim = 8 (obviously)
        lb_L = np.array([.1, .1, .05, .41, .0276, .081, .25, .0833])
        ub_L = np.array([.4, .4, .2, 1, .1702, .21, .5, .7])
        x_L = lb_L + (ub_L - lb_L)*np.array(x)
        f = R0(x_L)
        df = R0_grad(x_L)
        df *= (ub_L - lb_L)

    elif example == 12:
        # f = x[:,0]**2 - x[:,1]**2
        # df = npm.repmat(np.expand_dims(f,axis=1),1,dim)
        # df[:,0] = 2*x[:,0]
        # df[:,1] = -2*x[:,1]
        # f = (x[:,0]-0.)**2 + (x[:,1]-0.)**2
        # df = npm.repmat(np.expand_dims(f,axis=1),1,dim)
        # df[:,0] = 2*(x[:,0]-0.)
        # df[:,1] = 2*(x[:,1]-0.)

        f = np.sum(x**2, axis=1)
        df = 2*x


    elif example == 100:
        center = 0.0
        f  = np.sin(np.sum((x-center)**2.0, axis=1))
        ff = np.cos(np.sum((x-center)**2.0, axis=1))
        df = 2.0 * (x-center) * npm.repmat(np.expand_dims(ff, axis=1),1,dim)

    else:
        print('Wrong example number!')

    f = torch.from_numpy(f).float()
    df = torch.from_numpy(df).float()

    return f, df
