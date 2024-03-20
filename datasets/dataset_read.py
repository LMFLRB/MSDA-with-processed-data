from copy import deepcopy as copy
import torchvision.transforms as transforms
from scipy.io import loadmat
from utils import join, exists

from .unaligned_data_loader import UnalignedDataLoader
from .digits_five import return_DigitsFive
from .dataset import gather_domains, split_domains

base_dir = 'G:/Data' #'E:/Shared/Data' # 

def return_OfficeCaltech10(domain, data_mode):
    from scipy.io import loadmat
    if domain in ["amazon", 'dslr', 'webcam', 'caltech']:
        data = loadmat(join(base_dir, 'OfficeCaltech10', data_mode, f'{domain}_{data_mode}.mat'))
        return dict(imgs= data['features'], labels=data['labels'])
    else:
        Warning(f"no module for data {domain}")

def load_msd_dataset(dataset="PACS", 
                 data_mode="folder",
                 sub_folder=None,
                 splitted=False,
                 domain_all=['photo', "art_painting", 'cartoon', 'sketch'], 
                 target='sketch', 
                 cache=True, 
                 scale=32, 
                 batch_size=64, 
                 max_size=float("inf"),
                 drop_last=True,
                 **kwargs):  
    
    reload = not hasattr(load_msd_dataset, "dataset") or load_msd_dataset.dataset!=dataset
    
    if reload: 
        # load datafor all domains to make dataloader once for reusement
        load_msd_dataset.dataset=dataset        
        if dataset == 'DigitsFive':
            # the datasets have been split into train/test
            Domain_all = {domain: return_DigitsFive(domain,
                                        data_mode=data_mode) for domain in domain_all}            
            D_train = [Domain_all[key]['train'] for key in domain_all]
            D_test  = [Domain_all[key]['test']  for key in domain_all]
            

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([scale,scale], antialias=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            Train_Dataset = gather_domains(in_dict=dict(domains=D_train),
                                        transform=transform,
                                        cache=cache,
                                        data_mode="data")

            Test_Dataset  = gather_domains(in_dict=dict(domains=D_test),
                                        transform=transform,
                                        cache=cache,
                                        data_mode="data")
            
            Train_Dataset, Val_Dataset = split_domains(Train_Dataset,[0.85])
        elif dataset == "OfficeCaltech10" and data_mode not in ["folder", None]:
            # data without train/test split
            Domain_all = {domain: return_OfficeCaltech10(domain,
                                        data_mode=data_mode) for domain in domain_all}
            D_train = [Domain_all[key] for key in domain_all]
            Data_all = gather_domains(in_dict=dict(domains=D_train),
                                        cache=cache,
                                        data_mode="data")
            Train_Dataset, Val_Dataset, Test_Dataset = split_domains(Data_all, [0.6,0.1]) 
        elif dataset in ["Office31", "OfficeHome", "DomainNet", "PACS", "VLCS"] or \
            (dataset=="OfficeCaltech10" and data_mode in ["folder", None]):
            transform = transforms.Compose([
                transforms.Resize([256,256], antialias=True),
                transforms.RandomCrop(scale),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])  
            if dataset=="DomainNet":
                # folder with given split for train and test in .txt files
                Train_Dataset = gather_domains(
                                        in_dict=dict(root_dir=join(base_dir, dataset),
                                                        domains=domain_all),
                                        transform=transform,
                                        cache=cache,
                                        data_mode="folder",
                                        split='train')
                Test_Dataset = gather_domains(
                                        in_dict=dict(root_dir=join(base_dir, dataset),
                                                        domains=domain_all),
                                        transform=transform,
                                        cache=cache,
                                        data_mode="folder",
                                        split='test')
                Train_Dataset, Val_Dataset = split_domains(Train_Dataset, [0.85]) 
            else:
                if splitted:
                    Train_Dataset = gather_domains(
                                        in_dict=dict(root_dir=join(base_dir, dataset),
                                                domains=domain_all,
                                                sub_folder=sub_folder),
                                        data_mode=data_mode,
                                        transform=transform,
                                        cache=cache,
                                        split="train")
                    Test_Dataset = gather_domains(
                                        in_dict=dict(root_dir=join(base_dir, dataset),
                                                domains=domain_all,
                                                sub_folder=sub_folder),
                                        data_mode=data_mode,
                                        transform=transform,
                                        cache=cache,
                                        split="test")
                    try:
                        Val_Dataset = gather_domains(
                                        in_dict=dict(root_dir=join(base_dir, dataset),
                                                domains=domain_all,
                                                sub_folder=sub_folder),
                                        data_mode=data_mode,
                                        transform=transform,
                                        cache=cache,
                                        split="val")            
                    except:
                        Train_Dataset, Val_Dataset = split_domains(Train_Dataset, [0.85]) 
                        pass
                        
                else:# folder without train/test split                                             
                    Domain_all = gather_domains(
                                        in_dict=dict(root_dir=join(base_dir, dataset),
                                                domains=domain_all,
                                                sub_folder = "raw" if dataset=="OfficeCaltech10" else sub_folder),
                                        data_mode=data_mode,
                                        transform=transform,
                                        cache=cache,
                                        split=None)
                    Train_Dataset, Val_Dataset, Test_Dataset = split_domains(Domain_all, [0.6,0.1]) 
        
        load_msd_dataset.splits=[{domain: Dataset for domain, Dataset in zip(domain_all, split)} 
                                   for split in [Train_Dataset,Val_Dataset,Test_Dataset]]

    # reorder the target domain to the last position
    splits = [[split[domain] for domain in domain_all if domain != target]+[split[target]] 
              for split in load_msd_dataset.splits] 
    # construct dataloader
    return_data = [UnalignedDataLoader(split, 
                                       batch_size=batch_size, 
                                       max_size=max_size,
                                       drop_last=drop_last) for split in splits]
   
    return tuple(return_data)

def load_msd_processed(dataset="PACS",
        domain_all=['photo', "art_painting", 'cartoon', 'sketch'], 
        target='sketch', 
        cache=True, 
        batch_size=64, 
        max_size=float("inf"),
        resnet_type:str=None,
        feature_type:str="linear",
        **kwargs):
    
    # load data at the first task of a dataset
    if not hasattr(load_msd_processed, "dataset") or load_msd_processed.dataset!=dataset:        
        load_msd_dataset.dataset=dataset
        def load_splitted(dataset, base_dir, split, domain_all):
            data_path=join(base_dir, 
                           f"{dataset}-processed", 
                           f'{resnet_type}-{feature_type}', 
                           split)
            splitted = {}
            for domain in domain_all:
                data = loadmat(join(data_path, f"{domain}.mat"))
                splitted[domain] = dict(imgs=data['features'], labels=data['labels'].squeeze())
            return splitted

        load_msd_processed.splits = [load_splitted(dataset, base_dir, split, domain_all) 
                                     for split in ["train", "val", "test"]]  
    
    # reorder target domain to the last
    splits = [[split[domain] for domain in domain_all 
                if domain!=target]+[split[target]] 
                for split in load_msd_processed.splits]
    # construct dataloader
    Datasets = [UnalignedDataLoader(gather_domains(
                                            dict(domains=split),
                                            cache=cache,
                                            data_mode="data"),
                                    batch_size=batch_size, 
                                    max_size=max_size)
                        for split in splits]
    
    return tuple(Datasets)
    
def load_dataset(params, 
                 with_processed_data=True,
                 resnet_type:str=None,
                 feature_type:str="linear",
                 **kwargs):
    if with_processed_data:
        root_path=join(base_dir, 
                       f"{params['dataset']}-processed", 
                       f'{resnet_type}-{feature_type}', 
                        )
        if not exists(join(root_path, 'done')):
            from models import ImageProcess
            params_ = copy(params)
            params_['batch_size'] = 256
            params_['drop_last'] = False
            DataProcess = ImageProcess(dataset=params_['dataset'],
                                       cache=params_['cache'],
                                       domain_all=params_['domain_all'],
                                       root_dir=base_dir,
                                       resnet_type=resnet_type,
                                       feature_type=feature_type)
            to_process = {split: data for split, data in 
                          zip(["train","val", "test"], 
                              load_msd_dataset(**params_))}
            DataProcess(to_process)

        return load_msd_processed(**params,
                                    resnet_type=resnet_type,
                                    feature_type=feature_type)
    else:
        return load_msd_dataset(**params)