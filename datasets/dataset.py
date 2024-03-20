from __future__ import print_function
import torch.utils.data as data
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
from os.path import join, sep
import numpy as np
import torch

def gather_domains(in_dict={},
                 data_mode="data",
                 transform=None,
                 target_transform=None,
                 cache=True,
                 split=None):
    if data_mode=="folder":
        domains = [myImgFolder(join(in_dict['root_dir'], 
                                            domain if in_dict.get('sub_folder') is None else join(in_dict['sub_folder'], domain)
                                        ),
                                    transform=transform,
                                    target_transform=target_transform,
                                    cache=cache,
                                    split=split
                                    ) for domain in in_dict['domains']
                        ]
    else:
        domains = [myDataset(domain['imgs'], 
                            domain['labels'],
                            transform=transform,
                            target_transform=target_transform,
                            cache=cache) for domain in in_dict['domains']
                        ]
        
    return domains

# def split_domains(domains, percentage:list[float,]=[0.7]):
#     from torch.utils.data import random_split
#     n_splits = len(percentage)+1 if sum(percentage)<1 else len(percentage)
#     splits=[[None]*len(domains)]*n_splits
#     for n_d, domain in enumerate(domains):
#         domain_size=len(domain)
#         splits_size = [int(percent*domain_size) for percent in percentage]
#         if sum(percentage)<1:
#             splits_size.append(domain_size-sum(splits_size))
#         else:
#             splits_size[-1] = domain_size-sum(splits_size[:-1])
#         domain_splits = random_split(domain, splits_size)
#         for n_s, domain_split in enumerate(domain_splits):
#             splits[n_s][n_d]=domain_split
#     return tuple(splits)

def split_domains(domains, percentage:list[float,]=[0.7]):
    from torch.utils.data import random_split
    n_splits  = len(percentage)+1 if sum(percentage)<1 else len(percentage)
    n_domains = len(domains)

    def get_splits_size(domain, percentage):
        domain_size=len(domain)
        splits_size = [int(percent*domain_size) for percent in percentage]
        if sum(percentage)<1:
            splits_size.append(domain_size-sum(splits_size))
        else:
            splits_size[-1] = domain_size-sum(splits_size[:-1])
        return splits_size
    
    splits = [random_split(domain, get_splits_size(domain, percentage)) for domain in domains]
    splits = [[splits[n_d][n_s] for n_d in range(n_domains)] for n_s in range(n_splits)]
    return tuple(splits)

class myDataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, data, label,
                 transform=None,
                 target_transform=None,
                 cache=True):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label
        self.is_cache=cache
        self.device = torch.device("cuda:0")
        if cache:
            self.cached = dict()
        self.feature_shape=self.__getitem__(0)[0].shape

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """
        def load_img(index):
            img, target = self.data[index], self.labels[index]
            if len(img.shape)==1:
                img = torch.from_numpy(img).to(torch.float32)
                # target = torch.tensor([target], dtype=torch.long)
            else:
                if img.shape[0] != 1:
                    img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))            
                elif img.shape[0] == 1:
                    im = np.uint8(np.asarray(img))
                    im = np.vstack([im, im, im]).transpose((1, 2, 0))
                    img = Image.fromarray(im)
                                
                if self.transform is not None:
                    img = self.transform(img)
                else:
                    transform = transforms.ToTensor()
                    img = transform(img)

                if self.target_transform is not None:
                    target = self.target_transform(target)
                # else:
                #     target = torch.tensor([target], dtype=torch.long)

            return img, target        
        
        if self.is_cache:
            if index not in self.cached:
                img, target = load_img(index)
                self.cached[index] = list([img, target])
                # self.cached[index][0] = self.cached[index][0].cuda(non_blocking=True)
                # self.cached[index][1] = self.cached[index][1].cuda(non_blocking=True) \
                #                     if type(target)=='torch.Tensor' else \
                #             torch.tensor(self.cached[index][1]).cuda(non_blocking=True)
                
                
                self.cached[index][0] = self.cached[index][0].to(self.device)
                self.cached[index][1] = self.cached[index][1].to(self.device) \
                                    if isinstance(target, torch.Tensor) else \
                            torch.tensor(self.cached[index][1]).to(self.device)
            return tuple(self.cached[index])
        else:        
            return load_img(index)

    def __len__(self):
        return len(self.data)

class myImgFolder(data.Dataset):
    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 cache=True,
                 split=None):
        super(myImgFolder, self).__init__()
        self.dataset = ImageFolder(root=root,
                           transform=transform,
                           target_transform=target_transform)        
        self.is_cache=cache
        if cache:
            self.cached = dict()
        self.device = torch.device("cuda:0")

        domain = root.split(sep)[-1]
        self.name = domain
        if split is not None:
            samples, targets = [], []
            with open(join(root, f'{domain}_{split}.txt'), 'r') as f:
                for line in f:
                    filepath, label = line.strip().split()
                    samples.append((join(root,filepath.split('/',1)[1]), int(label)))
                    targets.append(int(label))
            self.dataset.imgs = samples
            self.dataset.samples = samples
            self.dataset.targets = targets
        self.feature_shape=self.dataset[0][0].shape

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """
        if self.is_cache:
            if index not in self.cached:
                img, target = self.dataset[index]
                self.cached[index] = list([img, target])
                # self.cached[index][0] = self.cached[index][0].cuda(non_blocking=True)
                # self.cached[index][1] = self.cached[index][1].cuda(non_blocking=True) \
                #                     if type(target)=='torch.Tensor' else \
                #             torch.tensor(self.cached[index][1]).cuda(non_blocking=True)
                
                self.cached[index][0] = self.cached[index][0].to(self.device)
                self.cached[index][1] = self.cached[index][1].to(self.device) \
                                    if type(target)=='torch.Tensor' else \
                            torch.tensor(self.cached[index][1]).to(self.device)
            return tuple(self.cached[index])
        else:        
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)