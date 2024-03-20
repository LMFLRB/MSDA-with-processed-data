from __future__ import print_function
import torch,yaml,argparse
import pandas as pd
from os.path import join,exists

from solver import Solver
from utils import transform_to_edict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Office31', help='dataset for experiment')
parser.add_argument('--model_type', type=str, default='M3SDA', help='model employed in the experiment')
parser.add_argument('--use_resnet', type=bool, default=True, help='use resnet in m3sda model to preprocess image data or not')
parser.add_argument('--train_resnet', type=bool, default=False, 
                    help='chosse to use resnet as prepocessNet with no grad or sharedNet in m3sda model')
parser.add_argument('--loss_to_use', type=list, default=['gjrd'], #'gcsd', 
                    help='losses to run the model')

parser.add_argument('--loss_configs', type=dict, default=dict(
                        moment={'msda_type': 'moment', 'msda_weight': 0.0005, 'params': 5},
                        mmd={'msda_type': 'mmd', 'msda_weight': 10, 'params': [1,2,5,10]},
                        gcsd={'msda_type': 'gcsd', 'msda_weight': 0.5,},
                        gjrd={'msda_type': 'gjrd', 'msda_weight': 0.5, 'params': 2},
                        # gjrd={'msda_type': 'gjrd', 'msda_weight': 0.5, 'params': 2, 'weights': [0.5,0.5]},
                        ), 
                    help='default configs for all the loss candidates')
parser.add_argument('--log_txt', type=bool, default=False, help='flag to log txt files or not')
parser.add_argument('--write_logs', type=bool, default=False, help='flag to log .event files or not')
parser.add_argument('--eval_once', type=bool, default=False, help='eval once or not')
parser.add_argument('--num_runs', type=int, default=10, help='unmber of montecarlo runs')
args = parser.parse_args()


def load_configs_from_args(configs):      
    configs.dataset = args.dataset
    configs.model_type = args.model_type
    configs.use_resnet = args.use_resnet
    configs.train_resnet = args.train_resnet
    configs.scale = 32 if args.dataset=="DigitsFive" else 224   
    if configs.dataset=="DigitsFive": 
        configs.batch_size = 128
    elif configs.dataset in ['OfficeCaltech10', 'Office31', 'OfficeHome']:        
        configs.batch_size = 32
    elif configs.dataset=="DomainNet":
        configs.batch_size = 16
        configs.train_resnet = True
        configs.data_cached = False
    if args.eval_once:        
        configs.max_epoch = 1
        configs.max_batches = 1
        args.num_runs = 1
    return configs

def main(configs):    
    seeds = torch.randint(0,10000,[args.num_runs,])
    for loss_config in [args.loss_configs[loss] for loss in args.loss_to_use]:
        configs.loss_config = transform_to_edict(loss_config)
        solver = Solver(**configs) 
        record = join(solver.root_dir, solver.record_dir, f"{solver.experiment}.csv")
        for version, seed in enumerate(seeds):
            average=0.0
            df_dict=dict(seed=seed.item())
            for target in solver.domain_all:
                solver.load_data_and_model(target, seed=seed, version=version)
                solver.fit(patience=10)
                df_dict[solver.experiment_split] = solver.test_acc_best.item() \
                        if hasattr(solver.test_acc_best, 'item') else solver.test_acc_best  
                average = average+solver.test_acc_best.item()
            df_dict['average'] = average/len(solver.domain_all)
            
            if not exists(record) or version not in pd.read_csv(record)['version'].values:
                data_to_append = pd.DataFrame(df_dict, columns=df_dict.keys() if version==0 else None, index=[version])
                data_to_append.to_csv(record, mode='a', index_label="version", 
                            header=True if (version==0 or not exists(record)) else False)  

if __name__ == '__main__':
    with open(f'config_for_M3SDA.yaml', 'r') as file:
        configs = transform_to_edict(yaml.safe_load(file))    
    load_configs_from_args(configs)
    configs.cuda = not configs.no_cuda and torch.cuda.is_available()
    print('\n global configurations:\n', configs)
    main(configs)