from math import ceil
import torch
import torch.nn as nn
from utils import jacobian


class ForBlock(nn.Module):
    '''
    One layer of the forward RevNet.
    '''
    def __init__(self, dim, step_size):
        super().__init__()
        self.dim = dim
        self.step_size = step_size

        self.layers = nn.ModuleList()
        for i in range(2):
            # self.layers.append(nn.Linear(ceil(self.dim/2), self.dim))
            self.layers.append(nn.Linear(ceil(self.dim/2), 2*ceil(self.dim/2)))

        # self.reset_parameters()   # optional parameter reset

    def reset_parameters(self):
        for name, param in self.layers.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        u_old, v_old = torch.split(x, [ceil(self.dim/2), ceil(self.dim/2)], 1)

        u_mid = torch.tanh(self.layers[0](v_old)).unsqueeze(2)
        u_sig = torch.matmul(self.layers[0].weight.t(), u_mid).squeeze(2)
        u_new = u_old + self.step_size * u_sig

        v_mid = torch.tanh(self.layers[1](u_new)).unsqueeze(2)
        v_sig = torch.matmul(self.layers[1].weight.t(), v_mid).squeeze(2)
        v_new = v_old - self.step_size * v_sig

        return torch.cat((u_new, v_new), 1)


class InvBlock(nn.Module):
    '''
    One layer of the backward RevNet.
    '''
    def __init__(self, dim, step_size):
        super().__init__()
        self.dim = dim
        self.step_size = step_size

        self.layers = nn.ModuleList()
        for i in range(2):
            # self.layers.append(nn.Linear(ceil(self.dim/2), self.dim))
            self.layers.append(nn.Linear(ceil(self.dim/2), 2*ceil(self.dim/2)))

        # self.reset_parameters()   # optional parameter reset

    def reset_parameters(self):
        for name, param in self.layers.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        u_old, v_old = torch.split(x, [ceil(self.dim/2), ceil(self.dim/2)], 1)

        v_mid = torch.tanh(self.layers[1](u_old)).unsqueeze(2)
        v_sig = torch.matmul(self.layers[1].weight.t(), v_mid).squeeze(2)
        v_new = v_old + self.step_size * v_sig

        u_mid = torch.tanh(self.layers[0](v_new)).unsqueeze(2)
        u_sig = torch.matmul(self.layers[0].weight.t(), u_mid).squeeze(2)
        u_new = u_old - self.step_size * u_sig

        return torch.cat((u_new, v_new), 1)


class RevNet(nn.Module):
    '''
    Reversible Neural Network built from forward and backward blocks.
    '''
    def __init__(self, dim, step_size, num_blocks):
        super().__init__()

        self.forward_net = nn.ModuleList()
        self.inverse_net = nn.ModuleList()
        for i in range(num_blocks):
            self.forward_net.append(ForBlock(dim, step_size))
            self.inverse_net.append(InvBlock(dim, step_size))

        for i in range(num_blocks):
            j = num_blocks - (i + 1)
            for k in range(2):
                self.inverse_net[i].layers[k].weight = (
                    self.forward_net[j].layers[k].weight )
                self.inverse_net[i].layers[k].bias = (
                    self.forward_net[j].layers[k].bias )

    def forward(self, x):
        z = self.feed_forward(x)
        x_bar, jac = self.feed_back_compute_jac(z)

        return z, x_bar, jac

    def feed_forward(self, x):
        for i, layer in enumerate(self.forward_net):
            x = layer(x)
        return x

    def feed_back_compute_jac(self, z):
        x = torch.clone(z)
        for i, layer in enumerate(self.inverse_net):
            x = layer(x)

        jac = jacobian(x, z)
        # jac_norm = torch.sqrt(torch.sum(jac*jac,1))
        # jac_norm = torch.unsqueeze(jac_norm, 1)
        # jac = jac / jac_norm   # Normalization to compare with original NLL
        return x, jac


class RegNet(nn.Module):
    '''
    Standard feed-forward network for computing the low-dimensional regressions.
    '''
    def __init__(self, input_dim, hidden_layers, hidden_neurons):
        super().__init__()
        self.input_dim = input_dim

        self.reg_net = nn.ModuleList()
        self.reg_net.append(nn.Linear(input_dim, hidden_neurons))
        for idx in range(hidden_layers):
            self.reg_net.append(nn.Linear(hidden_neurons, hidden_neurons))
        self.reg_net.append(nn.Linear(hidden_neurons, 1))

        # self.reset_parameters()   # optional parameter reset

    def reset_parameters(self):
        for name, param in self.reg_net.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, z):
        out = z[:,:self.input_dim]
        for i, layer in enumerate(self.reg_net):
            if i != len(self.reg_net) - 1:
                out = torch.tanh(layer(out))
            else:
                out = layer(out)
        return out
