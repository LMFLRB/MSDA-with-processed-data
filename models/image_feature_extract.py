import os, torch
import torch.nn as nn
# import numpy as np
# from tqdm import tqdm
# from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torch import stack
from os.path import join as join
from scipy.io import savemat
from .base import *

class ImageProcess(nn.Module):
    name = 'ImageProcessor'
    patch_size=[224,224]
    resnet_dict=dict(MNIST='resnet18',FashionMNIST='resnet34',STL10='resnet50')
    def __init__(self,   
                dataset:str="",   
                domain_all:list=[], 
                root_dir:str="",           
                resnet_type:str=None,
                feature_type:str="linear",
                **kwargs
                ):
        super(ImageProcess, self).__init__()
        self.use_cuda=torch.cuda.is_available()
        self.root_dir=root_dir
        self.dataset=dataset
        self.domain_all=domain_all
        self.feature_type=feature_type
        if resnet_type is None:
            self.resnet_type=self.resnet_dict[dataset] if \
                dataset in ['MNIST','FashionMNIST','STL10'] else 'resnet50'
        else:
            self.resnet_type=resnet_type

        self.model = FeatureResnet(self.resnet_type.upper(),feature_type=="linear")
        
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
        self.transform = transforms.Resize(self.patch_size, antialias=True)
        self.flags={}
        self.data_path_proc = join(self.root_dir, 
                                   f"{self.dataset}-processed", 
                                   f'{self.resnet_type}-{self.feature_type}')
        os.makedirs(self.data_path_proc, exist_ok=True) 
    
    def forward(self, datasets_to_process:dict):
        if os.path.exists(os.path.join(self.data_path_proc, 'done')):
            for split, _ in datasets_to_process.items():
                self.flags[split] = True
            print(f"{self.dataset} has beed processed")
            return
        else:   
            for split, dataset in datasets_to_process.items():
                self.process(dataset, split)
            if not False in self.flags.values():
                with open(os.path.join(self.data_path_proc, 'done'), 'w') as f:
                    f.write('done')
    
    def process(self, dataset, split="train", silent:bool=True):
        try:  
            print(f"{self.dataset} pre-processing for {split}-set .......")
            os.makedirs(join(self.data_path_proc, split), exist_ok=True)     
            processed = [([], []) for _ in range(len(self.domain_all))]
            for features,labels in dataset:
                for n_domain, (feature, label) in enumerate(zip(features, labels)):
                    if not dataset.stops[n_domain]:
                        processed[n_domain][0].extend(list(self.extract(feature).detach()))
                        processed[n_domain][1].extend(list(label))

            for (features,labels), domain in zip(processed, self.domain_all):
                data_path_proc = join(self.data_path_proc, split, f"{domain}.mat")
                savemat(data_path_proc, dict(features=stack(features).cpu().numpy(),
                                             labels=stack(labels).cpu().numpy())
                        )
            self.flags[split]=True
            print(f"{self.dataset} pre-processing for {split}-set sucessed")
        except:
            self.flags[split]=False
            Warning(f"{self.dataset} pre-processing for {split}-set failed")
        

    def extract(self, x):    
        if len(x.shape)==3:
            x = x.unsqeeze(1)
        if x.shape[1]==1:
            x = x.repeat([1,3,1,1])
        if x.shape[-1]!=224:
            x = self.transform(x)
        return self.model(x)
    