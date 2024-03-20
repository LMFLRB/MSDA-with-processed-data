import numpy as np
import os
import torch
import random
from easydict import EasyDict as edict
from typing import Optional

from os.path import dirname, join, exists

def get_father_path(order=1):
    father=dirname(__file__)
    for i in range(order):
        father=dirname(father)
    return father

def get_file(file_dir, part_name: str=r"mypart", format: str=r".myformat"):
    # find files in file_dir with part_name or format style
    myfiles, dirs = [], []
    try:
        for root, dirs, files in os.walk(file_dir):  
            for file in files:  
                filename, fileformat = os.path.splitext(file)
                if not format == r".myformat" and not part_name==r"mypart":
                    if  fileformat == format and part_name in filename:  # find the files with assigned format and part name
                        myfiles.append(os.path.join(root, file))  
                else:
                    if  fileformat == format or part_name in filename:  # find the files with assigned format or part name
                        myfiles.append(os.path.join(root, file))  
    except:
        Warning("failed to get file with the give format or part name !")
    return myfiles, dirs if len(dirs)>0 else [file_dir]

def clear_directory(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    num_classes = max(labels_dense)- min(labels_dense) + 1
    labels_one_hot= (np.eye(num_classes)[labels_dense]).astype(np.float32)
    return labels_one_hot

def earlystop(check_value, patience:int=10, eps:float=1.e-3):
    if not hasattr(earlystop, "count") or ( 
        hasattr(earlystop, "errors") and len(earlystop.errors)!=patience+1):
        earlystop.count = 0
    if earlystop.count == 0:
        earlystop.errors=torch.tensor(1).to(check_value.dtype).repeat(patience+1)
        earlystop.check_values=torch.tensor(1).to(check_value.dtype).repeat(patience+1)
        earlystop.check_value=torch.tensor(0,device=check_value.device,dtype=check_value.dtype)
        earlystop.count_without_improvement = 0
        earlystop.best = torch.tensor(0.0)
    earlystop.count += 1
    earlystop.check_values[1:]=earlystop.check_values[:-1].clone()
    earlystop.check_values[0] =(check_value.item())
    earlystop.errors[1:]=earlystop.errors[:-1].clone()
    earlystop.errors[0] =(torch.tensor(check_value.item())-earlystop.check_value).abs()
    flag=False
    if earlystop.count>patience:
        flag = (earlystop.errors<eps*(check_value.abs().item())).all().item()
        if flag:
            earlystop.stop_message = f"earlystopped with too small improvement for {patience} epochs"

    earlystop.check_value=torch.tensor(check_value.item())
    if earlystop.best<check_value.item():
        earlystop.best=check_value.item()
        earlystop.count_without_improvement = 0
    else:
        earlystop.count_without_improvement += 1
    if earlystop.count_without_improvement>patience:
        earlystop.stop_message = f"earlystopped with no improvement for {patience} epochs"
        flag = True
    
    return flag

def expand_dict(mydict:dict) -> dict:
    ex_dict={}
    for key, value in mydict.items():
        if isinstance(value, dict):
            ex_dict=dict(ex_dict, **expand_dict(value))
        else:
            ex_dict[key] = value
    return ex_dict

def update_callback(writer, iteration: int, update_dict: dict):
    paths=os.path.normpath(writer.logdir).split(os.sep)
    for key, value in expand_dict(update_dict).items():
        writer.add_scalar(tag=f"{paths[-1]}/{key}", 
                        scalar_value=value, 
                        global_step=iteration, 
                        display_name=key)
        
def transform_to_edict(in_dict):
    in_dict=edict(in_dict)
    for key, value in in_dict.items():
        if isinstance(value, dict):
            in_dict[key]=transform_to_edict(value)
    return in_dict  

def manual_seed_all(seed: Optional[int] = None, workers: bool = False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set the random seed for numpy
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"