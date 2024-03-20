from __future__ import print_function
import torch,sys,yaml
sys.path.append('./model')
sys.path.append('./datasets')
sys.path.append('./metric')

from solver import Solver
from utils import transform_to_edict


moment_match={'msda_type': 'moment', 'msda_weight': 0.0005, 'belta_moment': 5}
mmd_match={'msda_type': 'mmd', 'msda_weight': 10, 'sigma': [1,2,5,10]}
gcsd_match={'msda_type': 'gcsd', 'msda_weight': 0.5,}
gjrd_match={'msda_type': 'gjrd', 'msda_weight': 0.5, 'order': 2}

def main():    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    solver = Solver(**args) 
    for loss_config in [moment_match,mmd_match,gcsd_match,gjrd_match]:
    # for loss_config in [gcsd_match,gjrd_match]:
        solver.loss_config = loss_config
        for version in range(1,10):
            for target in solver.domain_all:
                solver.load_data_and_model(target, seed=1, version=version)
                if args.eval_only:
                    solver.test()
                else:
                    solver.fit()
                try:
                    solver.events_to_mat()
                except:
                    Warning('events_to_mat failed!')
                solver.writer.close()

if __name__ == '__main__':
    with open(f'config.yaml', 'r') as file:
        args = transform_to_edict(yaml.safe_load(file))
    print('\nglobal configurations:\n',args)
    main()