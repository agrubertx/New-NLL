import os, shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import os, shutil


def save_checkpoints(state, is_best=None,
                     base_dir='checkpoints',
                     save_dir=None):
    '''
    Saves network data during training.  Creates a 'checkpoints' directory if
    one is not already present.
    '''
    if save_dir:
        save_dir = os.path.join(base_dir, save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    checkpoint = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint)
    if is_best:
        best_model = os.path.join(save_dir, 'best_model.pth.tar')
        shutil.copyfile(checkpoint, best_model)


def jacobian(output, input):
    n, dim = input.shape
    w = torch.ones_like(input[:,[0]])

    jac = torch.empty(dim,n,dim).to(output.device)
    for i in range(dim):
        output_i = output[:,[i]]
        jac[i] = torch.autograd.grad(output_i, input, w, create_graph=True)[0]
    jac = jac.permute(1,0,2)
    return jac


def grad(outputs, inputs):
    """
    compute the derivative of outputs associated with inputs

    Params
    ======
    outputs: (N, 1) tensor
    inputs: (N, D) tensor
    """
    w = torch.ones_like(outputs)
    df = torch.autograd.grad(outputs, inputs,
                        grad_outputs=w,
                        create_graph=True)[0]
    return df


def normalize_data(data):
    max = np.max(data, 0)
    min = np.min(data, 0)
    normalized_data = (data-min) / (max-min)
    return normalized_data, max-min, min


def unnormalize_data(normalized_data, mm, min):
    unnormalized_data = normalized_data * mm + min
    return unnormalized_data


def plot_quiver(x, df, dh=None):
    x_numpy = x.detach().numpy()
    df_numpy = df.detach().numpy()
    # dg_numpy = dg.detach().cpu().numpy()
    if dh is not None:
        dh_numpy = dh.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))

    plt.quiver(x_numpy[:, 0], x_numpy[:, 1],
               df_numpy[:, 0], df_numpy[:, 1], color='red', width=0.0025)
    plt.draw()

    if dh is not None:
        plt.quiver(x_numpy[:, 0], x_numpy[:, 1],
                   dh_numpy[:, 0], dh_numpy[:, 1],
                   color='0.6', width=0.0025)
        plt.draw()

    plt.axis('equal')
    return fig


def plot_regression(z, f, f_bar):
    index = np.argsort(z.detach().cpu().numpy(), axis=0)
    znp = z.detach().numpy()
    fbnp = f_bar.detach().numpy()

    fig2 = plt.figure(1, figsize=(8, 8))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.xaxis.label.set_size(24)
    ax2.yaxis.label.set_size(24)
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.set_xlabel('Value of $z_1$', labelpad=15)
    ax2.set_ylabel('Function Value', labelpad=15)

    ax2.plot(znp[index,0][::10], f[index][::10],'*', color='#fc8d62',
             label='Exact', markersize=15)
    ax2.plot(znp[index,0][::10], fbnp[index][::10],'.', color='#66c2a5',
             label='Predicted', markersize=15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), prop={'size': 28})
    # plt.savefig('plot1d.pdf', format='pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()


# For global polynomial fit.
def fitFunc(x, a, b, c, d, e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4


# For local polynomial fit.
def LfitFunc(x, a, b, c):
    return a + b*x + + c*x**2


#basic reproduction number for Ebola example
def R0(x):
    b1 = x[:,0]; b2 = x[:,1]; b3 = x[:,2]; r1 = x[:,3]
    g1 = x[:,4]; g2 = x[:,5]; om = x[:,6]; p = x[:,7]

    return (b1 + b2*r1*g1/om + b3*p/g2)/(g1 + p)


# R0 gradient with respect to UNnormalized parameters
def R0_grad(x):
    b1 = x[:,0]; b2 = x[:,1]; b3 = x[:,2]; r1 = x[:,3]
    g1 = x[:,4]; g2 = x[:,5]; om = x[:,6]; p = x[:,7]

    dRdb1 = (1./(g1 + p))[:,None]
    dRdb2 = (r1*g1/om/(g1 + p))[:,None]
    dRdb3 = (p/g2/(g1 + p))[:,None]
    dRdr1 = (b2*g1/om/(g1 + p))[:,None]
    dRdg1 = (b2*r1/om/(g1 + p) - R0(x)/(g1 + p))[:,None]
    dRdg2 = (-b3*p/g2**2/(g1 + p))[:,None]
    dRdom = (-b2*r1*g1/om**2/(g1 + p))[:,None]
    dRdp = (b3/g2/(g1 + p) - R0(x)/(g1 + p))[:,None]

    return np.hstack((dRdb1, dRdb2, dRdb3, dRdr1, dRdg1, dRdg2, dRdom, dRdp))
