import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from pyDOE import lhs   # for fancy sampling, otherwise use np.random
import scipy.optimize as syopt  # for polynomial least-squares
from sklearn.neighbors import NearestNeighbors  # for local least-squares

from functions import test_func
import utils
from utils import jacobian, save_checkpoints, plot_quiver
from networks import RevNet, RegNet


def nll_loss(jac, df):
    norm_df2 = torch.sum(df*df, 1)
    df = torch.unsqueeze(df, 2)
    jac_df = torch.sum(jac*df, 1) #Nsamples x dim
    jac_df[:,0] = 0
    # jac_df[:,1] = 0
    res = torch.sum(jac_df * jac_df, 1)
    return torch.mean(res)


def jac_loss(jac):
    n = jac.shape[0]
    d = jac.shape[1]

    colNorm2 = torch.sum(jac * jac, 1)
    colNorm = torch.sqrt(colNorm2)
    interm = torch.unsqueeze(colNorm, 1)
    colMtx = interm.expand(-1, jac.shape[1], -1)
    nJac = jac / colMtx

    jac_det = torch.empty(n).to(jac.device)
    for i in range(n):
         uuu,eee,vvv = torch.svd(nJac[i,:,:])
         jac_det[i] = torch.prod(eee)
         # jac_det[i] = torch.trace(nJac[i,:,:])
    return torch.nn.MSELoss()(jac_det, torch.ones_like(jac_det))
    # return torch.mean(jac_det)


class Trainer():
    '''
    Contains several functions for training and testing the networks involved
    in the algorithm.
    '''
    def __init__(self, args):
        self.args = args
        self.dim = args.dim
        self.device = self.args.device

        PDE = 'PDE'  ## only for naming models
        self.hack = args.function if args.use_data==False else PDE

        # Define the NLL and Regression networks
        self.RevNet_name = (f'RevNet_{self.hack}_dim{args.dim}'
            + f'_N{args.train_num}_{args.num_blocks}RevBlocks' + f'_{args.hidden_layers}RegLayers_{args.hidden_neurons}Neurons')

        self.RegNet_name = (f'RegNet{self.hack}_dim{args.dim}' +
            f'_N{args.train_num}_{args.num_blocks}RevBlocks' + f'_{args.hidden_layers}RegLayers_{args.hidden_neurons}Neurons')

        self.RevNet = RevNet(args.dim, args.step_size,
                                args.num_blocks).to(args.device)

        self.RegNet = RegNet(args.reg_dim, args.hidden_layers,
                                args.hidden_neurons).to(args.device)

        print(f'The example number is {args.function}, the dimension' +
                f' is {args.dim}, and the regularizer weight' +
                f' is {args.jac_weight}')

        print(self.RevNet)
        print(self.RegNet)

        if args.use_data == True:
            # Generate training/validation data (testing == validation here)
            self.preprocess_data(args.train_num, args.valid_num)
        else:
            # Generate the training/validation/testing data
            self.xTrain = self.sample_domain(args.train_num)
            self.xTrain.requires_grad = True
            fTrain, dfTrain = test_func(self.xTrain, args.function)
            self.fTrain = fTrain.to(args.device)
            self.dfTrain = dfTrain.to(args.device)

            self.xValid = self.sample_domain(args.valid_num)
            fValid, dfValid = test_func(self.xValid, args.function)
            self.fValid = fValid.to(args.device)
            self.dfValid = dfValid.to(args.device)

            test_num = self.args.valid_num   ## or whatever
            self.xTest = self.sample_domain(test_num)
            fTest, dfTest = test_func(self.xTest, self.args.function)
            self.fTest = fTest.to(self.device)
            self.dfTest = dfTest.to(self.device)


    def sample_domain(self, num_samples):
        lb = np.zeros(self.dim)
        # lb = -1*np.ones(self.Npar)
        ub = np.ones(self.dim)
        # data = np.random.rand(num_samples, self.dim)
        data = lb + (ub - lb) * lhs(self.dim, num_samples)
        dataTensor = torch.from_numpy(data).float().to(self.device)
        return dataTensor


    def preprocess_data(self, train_num, valid_num):
        path = self.args.data_path
        data = np.loadtxt(path)

        # normalize training data to [0,1]
        train_data = data[:train_num, :]
        train_inputs = train_data[:, :self.dim]
        train_fs = train_data[:, self.dim]
        train_dfs = train_data[:, 1+self.dim:]

        train_Ninputs, train_inputs_mm, train_inputs_min = (
                utils.normalize_data(train_inputs) )
        train_Nfs, train_fs_mm, train_fs_min = (
                utils.normalize_data(train_fs) )
        train_Ndfs = train_inputs_mm / train_fs_mm * train_dfs

        # normalize validation data based on training data
        valid_data = data[-valid_num:, :]
        valid_inputs = valid_data[:, :self.dim]
        valid_fs = valid_data[:, self.dim]
        valid_dfs = valid_data[:, 1+self.dim:]

        valid_Ninputs = (valid_inputs - train_inputs_min) / train_inputs_mm
        valid_Nfs = (valid_fs - train_fs_min) / train_fs_mm
        valid_Ndfs = train_inputs_mm / train_fs_mm * valid_dfs

        # Keep track of parameters so we can unnormalize later
        self.f_mm_orig = train_fs_mm
        self.f_min_orig = train_fs_min

        # Add a zero column in case of odd dimension
        if self.dim % 2 == 1:
            fakeCol = np.zeros_like(train_Ninputs[:,[0]])
            train_Ninputs = np.concatenate( (train_Ninputs, fakeCol), 1)
            train_Ndfs = np.concatenate( (train_Ndfs, fakeCol), 1)
            valid_Ninputs = np.concatenate( (valid_Ninputs, fakeCol), 1)
            valid_Ndfs = np.concatenate( (valid_Ndfs, fakeCol), 1)

        self.xTrain = torch.from_numpy(train_Ninputs).float().to(self.device)
        self.xTrain.requires_grad = True
        self.fTrain = torch.from_numpy(train_Nfs).float().to(self.device)
        self.dfTrain = torch.from_numpy(train_Ndfs).float().to(self.device)

        self.xValid = torch.from_numpy(valid_Ninputs).float().to(self.device)
        self.fValid = torch.from_numpy(valid_Nfs).float().to(self.device)
        self.dfValid = torch.from_numpy(valid_Ndfs).float().to(self.device)

        self.xTest = self.xValid
        self.fTest = self.fValid
        self.dfTest = self.dfValid


    def train_nll(self):
        optimizer = torch.optim.Adam(self.RevNet.parameters(),
                                          lr=self.args.lr)
        best_loss=1e10
        tt=time.time()

        print('Training the NLL Network...')

        self.RevNet.train()
        for epoch in range(self.args.epochs_nll + 1):
            optimizer.zero_grad()

            x_bar, jac = self.RevNet(self.xTrain)[1:]

            trainLoss = nll_loss(jac, self.dfTrain)
            if self.args.jac_weight != 0:
                loss2 = jac_loss(jac)
                trainLoss += self.args.jac_weight * loss2

            trainLoss.backward()
            optimizer.step()

            # Validation Loop
            if (epoch)%100 == 0:
                self.RevNet.eval()
                x_bar, jac = self.RevNet(self.xValid)[1:]

                loss1 = nll_loss(jac, self.dfValid)
                loss2 = jac_loss(jac)

                validLoss = nll_loss(jac, self.dfValid)
                if self.args.jac_weight != 0:
                    loss2 = jac_loss(jac)
                    validLoss += self.args.jac_weight * loss2

                print(f'Epoch # {epoch:5d}: ' +
                      f'validation loss = {validLoss.item():.3e}, ' +
                      f'training loss = {trainLoss.item():.3e}, ' +
                      f'time = {time.time()-tt:.2f}s')

                is_best = validLoss.item() < best_loss
                if is_best:
                    best_loss = validLoss.item()
                    print('new best!')
                state = {
                    'epoch': epoch,
                    'state_dict': self.RevNet.state_dict(),
                    'best_loss': best_loss
                }
                save_checkpoints(state, is_best, save_dir=self.RevNet_name)
                tt = time.time()

        print('Training Finished!')


    def test_nll(self):
        self.RevNet.eval()
        best_model = torch.load(f'checkpoints/{self.RevNet_name}' +
                        '/best_model.pth.tar')
        self.RevNet.load_state_dict(best_model['state_dict'])

        z, x_bar, jac = self.RevNet(self.xTest)

        dfdx = torch.sum(self.dfTest * self.xTest, 1)
        avg_dfdx = torch.sum(self.dfTest, 0) / len(self.dfTest)

        dfdz = torch.matmul(self.dfTest.unsqueeze(1), jac).squeeze(1)
        avg_dfdz = torch.sum(dfdz, 0) / len(self.xTest)

        sens_orig = torch.abs(avg_dfdx) / torch.sum(torch.abs(avg_dfdx)) * 100
        sens_new = torch.abs(avg_dfdz) / torch.sum(torch.abs(avg_dfdz)) * 100

        print_old = [f'{x.item():.3f}%' for x in sens_orig]
        print_new = [f'{x.item():.3f}%' for x in sens_new]

        print(f'sensitivities in x are {print_old}')
        print(f'sensitivities in z are {print_new}')

        # Plot of active direction against first inactive direction
        fig = plot_quiver(self.xTest, self.dfTest, jac[:, :, 1])
        plt.show()


    def train_test_global_polynomial_reg(self):
        self.RevNet.eval()
        best_model_NLL = torch.load(f'checkpoints/{self.RevNet_name}' +
                                    '/best_model.pth.tar')
        self.RevNet.load_state_dict(best_model_NLL['state_dict'])

        zTrain = self.RevNet(self.xTrain)[0].to(self.device).detach()

        params = syopt.curve_fit(utils.fitFunc,
                    zTrain.numpy()[:,0], self.fTrain.numpy())[0]

        zTest = self.RevNet(self.xTest)[0].to(self.device)

        fTilde = utils.fitFunc(zTest.detach().numpy()[:,0], *params)
        fTilde = torch.from_numpy(fTilde).float().to(self.device)

        if self.args.use_data == True:
            self.fTest = utils.unnormalize_data(self.fTest,
                            self.f_mm_orig, self.f_min_orig)
            fTilde = utils.unnormalize_data(fTilde,
                            self.f_mm_orig, self.f_min_orig)

        MSE = torch.nn.MSELoss()(self.fTest, fTilde)
        NRMSE = torch.sqrt(MSE) / (torch.max(self.fTest)
                                    - torch.min(self.fTest)) * 100
        relL1 = ( torch.nn.L1Loss()(self.fTest, fTilde) /
                  torch.nn.L1Loss()(self.fTest, torch.zeros_like(self.fTest))
                    * 100 )
        relL2 = ( torch.sqrt(torch.nn.MSELoss()(self.fTest, fTilde) /
                torch.nn.MSELoss()(self.fTest,torch.zeros_like(self.fTest)))
                    * 100 )

        print(f'The normalized RMSE error is {NRMSE.detach().numpy():.5f}%')
        print(f'The relative L-1 Error is {relL1.detach().numpy():.5f}%')
        print(f'The relative L-2 Error is {relL2.detach().numpy():.5f}%')

        index = np.argsort(zTest.detach().cpu().numpy(), axis=0)

        fig2 = plt.figure(1, figsize=(8, 8))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.xaxis.label.set_size(24)
        ax2.yaxis.label.set_size(24)
        ax2.tick_params(axis='x', labelsize=18)
        ax2.tick_params(axis='y', labelsize=18)
        ax2.cla()
        ax2.plot(zTest.detach().numpy()[index,0][::10],
                 self.fTest[index][::10],'*', color='#fc8d62',
                 label='Exact', markersize=15)
        ax2.plot(zTest.detach().numpy()[index,0][::10],
                 fTilde.detach().numpy()[index][::10],'.', color='#66c2a5',
                 label='Predicted', markersize=15)
        ax2.set_xlabel('Value of $z_1$', labelpad=15)
        ax2.set_ylabel('Function Value', labelpad=15)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), prop={'size': 28})
        # plt.savefig('plot1d.pdf', format='pdf')
        plt.draw()
        plt.show()


    def train_test_local_polynomial_reg(self):
        self.RevNet.eval()
        best_model_NLL = torch.load(f'checkpoints/{self.RevNet_name}' +
                                    '/best_model.pth.tar')
        self.RevNet.load_state_dict(best_model_NLL['state_dict'])

        zTest = self.RevNet(self.xTest)[0].to(self.device)
        zTrain = self.RevNet(self.xTrain)[0].to(self.device)
        z1 = zTest[:,0]
        zt1 = zTrain[:,0]

        nbrs = NearestNeighbors(n_neighbors=10)
        nbrs.fit(zt1.unsqueeze(1).detach().numpy())

        idx = nbrs.kneighbors(z1.unsqueeze(1).detach().numpy(),
                              return_distance=False)

        fTilde = np.zeros_like(self.fTest.detach().numpy())
        for i in range(len(self.xTest)):
            params = syopt.curve_fit(utils.LfitFunc, zt1.detach(
                        ).numpy()[idx[i]], self.fTrain.numpy()[idx[i]])[0]
            fTilde[i] = utils.LfitFunc(z1[i].detach().numpy(), *params)

        fTilde = torch.from_numpy(fTilde).float().to(self.device)

        if self.args.use_data == True:
            self.fTest = utils.unnormalize_data(self.fTest,
                            self.f_mm_orig, self.f_min_orig)
            fTilde = utils.unnormalize_data(fTilde,
                            self.f_mm_orig, self.f_min_orig)

        MSE = torch.nn.MSELoss()(self.fTest, fTilde)
        NRMSE = torch.sqrt(MSE) / (torch.max(self.fTest)
                                    - torch.min(self.fTest)) * 100
        relL1 = ( torch.nn.L1Loss()(self.fTest, fTilde) /
                  torch.nn.L1Loss()(self.fTest, torch.zeros_like(self.fTest))
                    * 100 )
        relL2 = ( torch.sqrt(torch.nn.MSELoss()(self.fTest, fTilde) /
                torch.nn.MSELoss()(self.fTest,torch.zeros_like(self.fTest)))
                    * 100 )

        print(f'The normalized RMSE error is {NRMSE.detach().numpy():.5f}%')
        print(f'The relative L-1 Error is {relL1.detach().numpy():.5f}%')
        print(f'The relative L-2 Error is {relL2.detach().numpy():.5f}%')

        index = np.argsort(zTest.detach().cpu().numpy(), axis=0)

        fig2 = plt.figure(1, figsize=(8, 8))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.xaxis.label.set_size(24)
        ax2.yaxis.label.set_size(24)
        ax2.tick_params(axis='x', labelsize=18)
        ax2.tick_params(axis='y', labelsize=18)
        ax2.cla()
        ax2.plot(zTest.detach().numpy()[index,0][::10],
                 self.fTest[index][::10],'*', color='#fc8d62',
                 label='Exact', markersize=15)
        ax2.plot(zTest.detach().numpy()[index,0][::10],
                 fTilde.detach().numpy()[index][::10],'.', color='#66c2a5',
                 label='Predicted', markersize=15)
        ax2.set_xlabel('Value of $z_1$', labelpad=15)
        ax2.set_ylabel('Function Value', labelpad=15)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), prop={'size': 28})
        # plt.savefig('plot1d.pdf', format='pdf')
        plt.draw()
        plt.show()


    def train_reg(self):
        self.RevNet.eval()
        best_model_NLL = torch.load(f'checkpoints/{self.RevNet_name}' +
                                    '/best_model.pth.tar')
        self.RevNet.load_state_dict(best_model_NLL['state_dict'])

        zTrain = self.RevNet(self.xTrain)[0].to(self.device)
        zValid = self.RevNet(self.xValid)[0].detach().to(self.device)

        optimizer = torch.optim.Adam(self.RegNet.parameters(),
                                          lr=self.args.lr)
        best_loss=1e10
        tt=time.time()

        print('Training the Regression Network...')

        self.RegNet.train()
        for epoch in range(self.args.epochs_reg + 1):
            optimizer.zero_grad()

            out = self.RegNet(zTrain)
            trainLoss = torch.nn.MSELoss()(self.fTrain, out.squeeze(1))

            trainLoss.backward(retain_graph=True)
            optimizer.step()

            # Validation Loop
            if (epoch)%100 == 0:
                with torch.no_grad():
                    out = self.RegNet(zValid)
                    validLoss = torch.nn.MSELoss()(self.fValid, out.squeeze(1))

                print(f'Epoch # {epoch:5d}: ' +
                      f'validation loss = {validLoss.item():.3e}, ' +
                      f'training loss = {trainLoss.item():.3e}, ' +
                      f'time = {time.time()-tt:.2f}s')

                is_best = validLoss.item() < best_loss
                if is_best:
                    best_loss = validLoss.item()
                    print('new best!')
                state = {
                    'epoch': epoch,
                    'state_dict': self.RegNet.state_dict(),
                    'best_loss': best_loss
                }
                save_checkpoints(state, is_best, save_dir=self.RegNet_name)
                tt = time.time()

        print('Training Finished!')


    def test_reg(self):
        self.RevNet.eval()
        best_model_NLL = torch.load(f'checkpoints/{self.RevNet_name}' +
                                    '/best_model.pth.tar')
        self.RevNet.load_state_dict(best_model_NLL['state_dict'])

        zTrain = self.RevNet(self.xTrain)[0].detach().to(self.device)
        zValid = self.RevNet(self.xValid)[0].detach().to(self.device)
        zTest = self.RevNet(self.xTest)[0].detach().to(self.device)

        self.RegNet.eval()
        best_model_reg = torch.load(f'checkpoints/{self.RegNet_name}' +
                                    '/best_model.pth.tar')
        self.RegNet.load_state_dict(best_model_reg['state_dict'])

        fTilde = self.RegNet(zTest).squeeze(1)

        if self.args.use_data == True:
            self.fTest = utils.unnormalize_data(self.fTest,
                            self.f_mm_orig, self.f_min_orig)
            fTilde = utils.unnormalize_data(fTilde,
                            self.f_mm_orig, self.f_min_orig)

        MSE = torch.nn.MSELoss()(self.fTest, fTilde)
        NRMSE = torch.sqrt(MSE) / (torch.max(self.fTest)
                                    - torch.min(self.fTest)) * 100
        relL1 = ( torch.nn.L1Loss()(self.fTest, fTilde) /
                  torch.nn.L1Loss()(self.fTest, torch.zeros_like(self.fTest))
                    * 100 )
        relL2 = ( torch.sqrt(torch.nn.MSELoss()(self.fTest, fTilde) /
                torch.nn.MSELoss()(self.fTest,torch.zeros_like(self.fTest)))
                    * 100 )

        print(f'The normalized RMSE error is {NRMSE.detach().numpy():.5f}%')
        print(f'The relative L-1 Error is {relL1.detach().numpy():.5f}%')
        print(f'The relative L-2 Error is {relL2.detach().numpy():.5f}%')

        index = np.argsort(zTest.detach().cpu().numpy(), axis=0)

        fig2 = plt.figure(1, figsize=(8, 8))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.xaxis.label.set_size(24)
        ax2.yaxis.label.set_size(24)
        ax2.tick_params(axis='x', labelsize=18)
        ax2.tick_params(axis='y', labelsize=18)
        ax2.cla()
        ax2.plot(zTest.detach().numpy()[index,0][::10],
                 self.fTest[index][::10],'*', color='#fc8d62',
                 label='Exact', markersize=15)
        ax2.plot(zTest.detach().numpy()[index,0][::10],
                 fTilde.detach().numpy()[index][::10],'.', color='#66c2a5',
                 label='Predicted', markersize=15)
        ax2.set_xlabel('Value of $z_1$', labelpad=15)
        ax2.set_ylabel('Function Value', labelpad=15)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), prop={'size': 28})
        # plt.savefig('plot1d.pdf', format='pdf')
        plt.draw()
        plt.show()
