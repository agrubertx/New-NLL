import argparse
import torch

class Options(object):
    '''
    This will be parsed before training.  Modify these options to change
    the hyperparameters of the algorithm.
    '''
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed',
                            type=int,
                            default=150,
                            help='Seed for pseudo-deterministic behavior')
        parser.add_argument('--use_data',
                            action='store_true',
                            default=False,
                            help='Bool to indicate if dataset should be used')
        parser.add_argument('--function',
                            type=int,
                            default=9,
                            help='Case in functions.py')
        parser.add_argument('--data_path',
                            type=str,
                            default='./data/burgersQ2wMu3_FDshuffled.dat',
                            help='Filepath to data (if --usedata = True)')
        parser.add_argument('--dim',
                            type=int,
                            default=2,
                            help='Dimension of input space')
        parser.add_argument('--num_blocks',
                            type=int,
                            default=10,
                            help='Number of layers in RevNet')
        parser.add_argument('--step_size',
                            type=float,
                            default=0.25,
                            help='Time-step scalar in RevNet')
        parser.add_argument('--reg_dim',
                            type=int,
                            default=1,
                            help='Number of active variables for regression')
        parser.add_argument('--hidden_layers',
                            type=int,
                            default=2,
                            help='Number of layers in RegNet')
        parser.add_argument('--hidden_neurons',
                            type=int,
                            default=20,
                            help='Number of neurons per layer in RegNet')
        parser.add_argument('--jac_weight',
                            type=float,
                            default=0,
                            help='Weight in regularizer term')
        parser.add_argument('--lr',
                            type=float,
                            default=0.0005,
                            help='Initial learning rate for ADAM optimizer')
        parser.add_argument('--epochs_nll',
                            type=int,
                            default=10000,
                            help='Number of epochs to train NLL')
        parser.add_argument('--epochs_reg',
                            type=int,
                            default=2000,
                            help='Number of epochs to train Regression Net')
        parser.add_argument('--train_num',
                            type=str,
                            default=500, #was 1000#
                            help='Number of Training data')
        parser.add_argument('--valid_num',
                            type=str,
                            default=500,
                            help='Number of Validating data')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        return args

if __name__ == '__main__':
    args = Options().parse()
    print(args)
