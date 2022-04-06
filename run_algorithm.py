import torch
from options import Options
from trainer2 import Trainer

'''
This script will run the NLL algorithm with the options specified in
options.py.  The 'trainer' object contains several ways to compute the low
dimensional regression, which can be commented in or out as desired.
'''

args = Options().parse()
torch.manual_seed(args.seed)
if args.device != 'cpu':
    torch.cuda.manual_seed(args.seed)

trainer = Trainer(args)

trainer.train_nll()
trainer.test_nll()

trainer.train_reg()
trainer.test_reg()

# trainer.train_test_local_polynomial_reg()
# trainer.train_test_global_polynomial_reg()
