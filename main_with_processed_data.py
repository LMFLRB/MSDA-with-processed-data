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

    

if __name__ == '__main__':
    with open(f'configs/config.yaml', 'r') as file:
        args = transform_to_edict(yaml.safe_load(file))
    print('\nglobal configurations:\n',args)
    args.cuda = torch.cuda.is_available()
    # args.cuda = False
    args.data_cache = args.cuda

    # # processing data with resnet first
    # for dataset in ['OfficeCaltech10', 'Office31', 'OfficeHome', 'PACS', 'VLCS', 'DigitsFive']:
    #     args.resnet_type = "resnet18" if dataset=='DigitsFive' else "resnet50"
    #     args.dataset = dataset
    #     solver = Solver(**args)
    #     solver.load_data_and_model(solver.domain_all[0], seed=1, version=0)
    #     print(solver.datasets.feature_shapes)
                
    # args.model_type = "M3SDA"
    args.model_type = "BCDA"
    # train msda tasks
    for dataset in ['DigitsFive', 'Office31', 'OfficeHome', 'PACS', 'VLCS']:
        args.resnet_type = "resnet18" if dataset=='DigitsFive' else "resnet50"
        args.batch_size = 256 if dataset=='DigitsFive' else 128
        args.dataset = dataset
        # for loss_config in [moment_match,mmd_match,gcsd_match,gjrd_match]:
        for loss_config in [gcsd_match]:
            args.loss_config.update(loss_config)
            solver = Solver(**args)
            for version in range(1,10):
                solver.load_data_and_model(solver.domain_all[0], seed=1, version=version)
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
                    if hasattr(solver, "writer"):
                        solver.writer.close()