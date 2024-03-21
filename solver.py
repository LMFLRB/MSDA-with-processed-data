from __future__ import print_function
import numpy as np
import torch
import os
import time
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from os.path import join, exists
from torch.autograd import Variable
from copy import deepcopy as copy
from typing import Union
from math import exp
from tensorboardX import SummaryWriter

from models import *
from metrics import *
from datasets import load_dataset
from utils import (update_callback, 
                   manual_seed_all, 
                   earlystop,
                   get_father_path,
                   clear_directory,
                   myEventLoader)

# logdir = join(get_father_path(4), 'logs')
logdir = join('G:/MyCode', 'Logs')
# Training settings
class Solver(object):
    def __init__(self, 
                 max_epoch:int=100,
                 max_batches=float("inf"),
                 batch_size:int=64,
                 learning_rate=0.0002, 
                 save_period:int=10,
                 optimizer:str='adam', 
                 data_cache:bool=True,
                 dataset:str='digit_five', 
                 scale:int=32,
                 model_type='M3SDA', 
                 loss_config:dict={},
                 num_classifier:int=2,
                 num_generator_update:int=4,
                 cuda:bool=True,
                 root_dir:Union[os.PathLike,str]='MSDA_logs',
                 checkpoint_dir:Union[os.PathLike,str]=None,
                 record_dir:Union[os.PathLike,str]=None,
                 resume_epoch:Union[int,str]='best',
                 test_period:int=1,
                 eval_only:bool=False,
                 test_model:str='best',
                 save_model:bool=True,
                 use_abs_diff:bool=False,
                 target:Union[str,int]=0,
                 ensemble_schema:str='acc_weighted',
                 enable_early_stop:bool=False,
                 log_txt=False,
                 write_logs=True,
                 clc_loss_type:str='NLL',
                 resume:bool=False,
                 use_resnet:bool=False,
                 train_resnet=False,
                 with_processed_data=True,
                 feature_type="linear",
                 resnet_type="resnet50",
                 **kwargs):
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.root_dir = join(logdir,root_dir)
        self.scale = scale


        self.checkpoint_dir = checkpoint_dir
        self.record_dir = record_dir
        self.save_model = save_model
        self.save_period = save_period
        self.use_abs_diff = use_abs_diff
        self.eval_only = eval_only
        self.resume = resume
        self.test_model = test_model
        self.resume_epoch = resume_epoch
        self.test_period = test_period
        self.max_epoch = max_epoch
        self.cuda = cuda
        self.cache = data_cache and cuda
        self.device = "cuda" if cuda else "cpu"

        self.loss_config=copy(loss_config)
        self.msda_type = self.loss_config.pop('msda_type')
        self.msda_crit = globals()[self.msda_type]
        
        self.cls_crit = ce_cls_loss if clc_loss_type.upper()=="CE" else nll_cls_loss

        self.optimizer = optimizer
        self.num_classifier = num_classifier
        self.num_generator_update = num_generator_update
           
        self.target = target
        self.ensemble_schema = ensemble_schema
        self.lr = learning_rate
        self.enable_early_stop = enable_early_stop
        
        self.dataset =  dataset
        self.model_type =  model_type
        self.log_txt = log_txt 
        
        self.Orig_write_logs = write_logs 
        self.write_logs = write_logs 
        self.use_resnet=use_resnet
        self.train_resnet=train_resnet

        self.with_processed_data = with_processed_data
        self.feature_type = feature_type
        self.resnet_type = resnet_type
        if with_processed_data:
            self.root_dir = join(self.root_dir,f"{resnet_type}-{feature_type}")

        if self.dataset=='DigitsFive':            
            self.domain_all = ['mnistm', 'mnist', 'usps', 'svhn', 'synthetic']
            self.domain_all_Abbreviation = ['mm', 'mt', 'up', 'sv', 'sy']
            self.num_class = 10
            print(self.domain_all)
        elif self.dataset=='OfficeHome':             
            self.domain_all = ["Art", 'Clipart', 'Product', 'Real World']
            self.domain_all_Abbreviation = ['A', 'C', 'P', 'R']
            self.num_class = 65
        elif self.dataset=='Office31':             
            self.domain_all = ["amazon", 'dslr', 'webcam']
            self.domain_all_Abbreviation = ['A', 'D', 'W']
            self.num_class = 31
        elif self.dataset=='OfficeCaltech10':          
            self.domain_all = ["amazon", 'dslr', 'webcam', 'caltech']
            self.domain_all_Abbreviation = ['A', 'D', 'W', 'C']
            self.num_class = 10
        elif self.dataset=='PACS':          
            self.domain_all = ["art_painting", 'cartoon', 'photo', 'sketch']
            self.domain_all_Abbreviation = ['A', 'C', 'P', 'S']
            self.num_class = 10
        elif self.dataset=='VLCS':          
            self.domain_all = ["CALTECH", 'LABELME', 'PASCAL', 'SUN']
            self.domain_all_Abbreviation = ['C', 'L', 'P', 'S']
            self.num_class = 10
        elif self.dataset=='DomainNet':          
            self.domain_all = ["clipart", 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
            self.domain_all_Abbreviation = ['C', 'I', 'P', 'Q', 'R', 'S']
            self.num_class = 345
        self.num_source = len(self.domain_all)-1

        self.experiment_dir = join(self.model_type, self.msda_type, self.dataset)   
        self.experiment = '-'.join(item for item in self.experiment_dir.split(os.sep))
        print('\nsolver initialization completed!\n')

    def load_data_and_model(self, target=int(0), version=None, seed=1, load_model='last'):   
        self.seed = seed
        self.epoch = 0       
        self.test_acc = torch.zeros([1,])    
        self.test_acc_best = torch.zeros([1,])
        self.val_acc = torch.zeros([1,])     
        self.val_acc_best = torch.zeros([1,])      

        self.target = target if isinstance(target, str) else self.domain_all[target]
        indx = target if isinstance(target, int) else self.domain_all.index(target)
        self.target_Abbrev = self.domain_all_Abbreviation[indx]
    
        self.sources = copy(self.domain_all_Abbreviation)
        self.sources.remove(self.domain_all_Abbreviation[indx])  
        self.experiment_split = ','.join(domain for domain in self.sources)+f"-to-{self.domain_all_Abbreviation[indx]}"
        self.record_name = join(self.experiment_dir, self.experiment_split)
        self.make_path(version)
          
        # manual_seed_all(self.seed)   
        print(f'dataset loading for {join(self.model_type, self.msda_type, self.dataset)}...')
        data_params=dict(dataset=self.dataset,
                         domain_all=self.domain_all,
                         target=self.target,
                         cache=self.cache,
                         batch_size=self.batch_size,
                         max_size=self.max_batches)
        self.datasets, self.dataset_val, self.dataset_test = load_dataset( 
                        data_params,
                        with_processed_data=self.with_processed_data,
                        resnet_type=self.resnet_type,
                        feature_type=self.feature_type)
        
        print('dataset loaded!\n')
        self.max_iter = (len(self.datasets)*self.max_epoch)
        self.max_iter_val = (len(self.dataset_val)*self.max_epoch)
        self.max_iter_test = (len(self.dataset_test)*self.max_epoch)
        self.verify_features()
        self.train_save_period = min(self.save_period, max(1,int(len(self.datasets)/3.0)))
        self.test_save_period = len(self.dataset_test)
        self.val_save_period = len(self.dataset_val)

        print(f'model loading for {join(self.model_type, self.msda_type, self.dataset)}...')
        self.model = globals()[self.model_type](
                                self.num_class, 
                                self.num_classifier if self.model_type=='M3SDA' else self.num_source, 
                                self.scale, 
                                is_image=self.is_image,
                                use_resnet=self.use_resnet,
                                feat_hiddens=self.feat_hiddens,
                                dim_input=self.dim_input,
                                resnet_type='ResNet50' if self.dataset=='office31' else 'ResNet101',
                                train_resnet=self.train_resnet,
                                with_processed_data=self.with_processed_data,
                                feature_type=self.feature_type,)

        # load model from ckpt
        self.early_stopped=False
        self.max_epoch=100
        if self.resume or self.eval_only:
            to_load = self.test_model if self.eval_only else (int(-1) if version is None else load_model)
            to_load = join(self.ckpt_dir, f"model_{to_load if isinstance(to_load,str) else f'epoch_{to_load}'}.ckpt")
            if exists(to_load):
                state_dict = torch.load(to_load)
                self.epoch = state_dict['epoch']
                self.max_epoch = self.epoch if (load_model=='last' and state_dict.get('early_stopped')==True) else self.max_epoch
                self.model.load_state_dict(state_dict['model'])    
                self.seed = state_dict['seed'] 
                self.test_acc_best = state_dict['test_acc_best'] if state_dict.get('test_acc_best') is not None else self.test_acc_best
        manual_seed_all(self.seed)    
        self.modelist = [self.model.sharedNets, self.model.clsFCs]
        if self.model_type=='MFSAN':
            self.modelist.append(self.model.sonNets)
        if self.model_type=='BCDA':
            self.modelist.append(self.model.fusionNet)
        if self.cache or self.cuda:
            self.model=self.model.cuda()

        if self.Orig_write_logs and self.epoch<self.max_epoch and not self.eval_only: 
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.EventLoader = myEventLoader(self.log_dir)
        else:
            self.write_logs=False
        self.set_optimizer()
        print('model loaded!\n')

    def make_path(self, version=None):
        if version == None:
            version = 0
            while exists(join(self.root_dir, self.record_dir, self.record_name, f"version_{version}")):
                version += 1  
        self.version = version
        self.record_name = join(self.record_name, f"version_{version}")
        self.log_dir = join(self.root_dir, self.record_dir, self.record_name)
        self.ckpt_dir= join(self.root_dir, self.checkpoint_dir, self.record_name)
        if exists(self.log_dir):
            if not (self.resume or self.eval_only):
                clear_directory(self.log_dir)
                clear_directory(self.ckpt_dir)
        if self.write_logs:
            os.makedirs(self.log_dir,exist_ok=True)
            self.record_train_csv = join(self.log_dir, f'train_results.csv')
            self.record_val_csv= join(self.log_dir, f'val_results.csv')
            self.record_test_csv= join(self.log_dir, f'test_results.csv')
        if self.save_model:
            os.makedirs(self.ckpt_dir,exist_ok=True)

    def verify_features(self):
        shape = self.dataset_test.feature_shapes[0]
        if len(shape)==3 and shape[-1]==shape[-2] and shape[0] in [1,3,2048,512]:
            self.is_image=True
            self.dim_input=shape[0]
            self.feat_hiddens = [512,256,128] if shape[0] in [1,3] \
                else ([1024,512,256] if shape[0]==2048 else [512,256,128])
        else:
            self.is_image=False
            self.dim_input=shape[-1]
            self.feat_hiddens = [500,500,1000]
    
    def events_to_mat(self):
        self.EventLoader.events_to_mat()

    def set_optimizer(self, momentum=0.9):
        if self.optimizer == 'SGD':
            kwargs=dict(lr=self.lr, weight_decay=0.0005,momentum=momentum) 
        elif self.optimizer == 'Adam':
            kwargs=dict(lr=self.lr, weight_decay=0.0005)
        if self.model_type=='BCDA':
            self.opt_gs = getattr(optim, self.optimizer)(self.model.sharedNets[0].parameters(), **kwargs)
            self.opt_gt= getattr(optim, self.optimizer)(self.model.sharedNets[1].parameters(), **kwargs)
            self.opt_c = getattr(optim, self.optimizer)(self.model.clsFCs.parameters(), **kwargs)
            self.opt_a = getattr(optim, self.optimizer)(self.model.fusionNet.parameters(), **kwargs)
            self.opt_all = [self.opt_c, self.opt_a, self.opt_gs, self.opt_gt]
        else:
            self.opt_gs = getattr(optim, self.optimizer)(self.model.sharedNets.parameters(), **kwargs)
            self.opt_c = [getattr(optim, self.optimizer)(cls.parameters(), **kwargs) for cls in self.model.clsFCs]  
            self.opt_all = [self.opt_gs]+self.opt_c
            if self.model_type=='MFSAN':
                self.opt_s = [getattr(optim, self.optimizer)(son.parameters(), **kwargs) for son in self.model.sonNets]             
                self.opt_all = self.opt_all+self.opt_s
        
    def step_optimizer(self, ):
        for opt in self.opt_all:
            opt.step()

    def reset_grad(self):
        for opt in self.opt_all:
            opt.zero_grad() 
 
    def loss_to_dict(self, loss_c, loss_msda, loss_dis):
        loss_dict = {f'loss_c{i+1}': loss_c[i].data.item() for i in range(len(loss_c))}
        loss_dict['loss_msda'] = loss_msda.data.item()
        loss_dict['dis_fix_g'] = loss_dis[0].data.item()
        if len(loss_dis)>1:
            loss_dict = dict(loss_dict, 
                    **{f'dis_fix_c_{i+1}': loss.data.item() for i,loss in enumerate(loss_dis[1:])})
        
        return loss_dict
    
    def lr_to_dict(self):
        # lr_dict = dict(lr_g=self.opt_gs.param_groups[0]["lr"],
        #                **{f'lr_c{i+1}': opt.param_groups[0]["lr"] for i,opt in enumerate(self.opt_c)})
        lr_dict = {f"lr_{i}": opt.param_groups[0]["lr"] for i,opt in enumerate(self.opt_all)}
        return lr_dict

    def log_update_train(self, loss_c, loss_msda, loss_dis, iteration, log_lr=True):
        update_dict = self.loss_to_dict(loss_c, loss_msda, loss_dis)
        if log_lr:
            update_dict.update(self.lr_to_dict())
        update_callback(self.writer, iteration, update_dict)
        return update_dict
    
    def log_update_train_BCDA(self, loss_c, loss_src_align, loss_tgt_align, iteration, log_lr=True):
        update_dict = dict(loss_c=loss_c.item(), 
                           loss_src_align=loss_src_align.item(), 
                           loss_tgt_align=loss_tgt_align.item())
        if log_lr:
            update_dict.update(self.lr_to_dict())
        update_callback(self.writer, iteration, update_dict)
        return update_dict

    def log_update_eval(self, acc, accs, iteration, eval_loss=None):
        eval_dict = dict(acc=acc.data.item(),
                         **{f'acc{i+1}': acc.data.item() for i,acc in enumerate(accs)})
        if eval_loss is not None:
            eval_dict['eval_loss'] = eval_loss.tolist()
        update_callback(self.writer, iteration, eval_dict)
        return eval_dict
    
    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))
    
    def forward_all(self, srcs, **kwargs):
        return self.model.forward_all(srcs, **kwargs)
    
    def forward(self, src, **kwargs):
        return self.model.forward(src, **kwargs)
    
    def loss_all_domain(self, feats, outputs, labels):   
        return  self.model.loss_all_domain(feats, outputs, labels, 
                                                       self.cls_crit, 
                                                       self.msda_crit, 
                                                       copy(self.loss_config))
    
    def loss_single_domain(self, fea_src, fea_tgts, pred_src, label_src, pred_tgts):        
        msda_params = copy(self.loss_config)  
        return self.model.loss_single_domain(fea_src, fea_tgts, pred_src, pred_tgts, 
                                             label_src, 
                                             self.cls_crit, 
                                             self.discrepancy, 
                                             self.msda_crit, 
                                             msda_params)
    
    def discrepancy(self, outs):
        if type(outs) in [tuple,list,torch.Tensor]:
            n=len(outs)
            # print(type(outs), n)
            if n==2:
                return (outs[0]-outs[1]).abs().mean()
            else:
                return sum([(outs[row] - outs[col]).abs().mean() for row,col in zip(*np.triu_indices(n,1))])
        else:
            Warning("Wrong input type.")

    def get_display_flag(self, batch_idx, mode="train"):
        if mode=="train":
            iteration = batch_idx+1+self.epoch*len(self.datasets)
            flag=(iteration<self.train_save_period or iteration % self.train_save_period == 0)
        elif mode=="test":
            iteration = batch_idx+1+(self.epoch-1)*len(self.dataset_test)
            flag=(iteration<self.test_save_period or iteration % self.test_save_period == 0)
        else:
            iteration = batch_idx+1+(self.epoch-1)*len(self.dataset_val)
            flag=(iteration<self.val_save_period or iteration % self.val_save_period == 0)

        return iteration, flag

    def train_BCDA(self):
        start_time = time.time()
        self.model.train()
        for batch_idx, (features,labels) in enumerate(self.datasets):     
            features = [Variable(feature).to(self.device) for feature in features]
            labels = [Variable(label.long()).to(self.device) for label in labels]  
            
            ### ********************************************************************************
            ### ***************************    procedure 1    **********************************
            ###     fix Gt, C, and train Gs,A to get barycenter of all srcs with minimizing BCD of
            ###          (z_1,\cdots,z_s,\bar{z})
            self.reset_grad()     
            for _ in range(self.num_generator_update):
                feats, outputs = self.forward_all(features)
                loss_src_bc = self.model.loss_src_align(feats[0], feats[-1], self.msda_crit, 
                                                        copy(self.loss_config))
                loss_src_bc.backward()
                self.opt_a.step()
                self.opt_gs.step()
                self.reset_grad()

            ### ********************************************************************************
            ### ***************************    procedure 2    **********************************
            ### fix Gs, A, C, and train Gt to align z_t to \bar{z} with their discrepency 
            for _ in range(self.num_generator_update):
                feats, outputs = self.forward_all(features)
                loss_tgt_bc = self.model.loss_tgt_align(feats[1], feats[-1], self.msda_crit, 
                                                        copy(self.loss_config))
                loss_tgt_bc.backward()
                self.opt_gt.step()
                self.reset_grad()

            ### ******************************************************************************** 
            ### ***************************    procedure 3    **********************************
            ###                             train all and mainly C
            feats, outputs = self.forward_all(features)
            loss_c, loss_src_align, loss_tgt_align = self.loss_all_domain(feats, outputs, labels[:-1])
            
            loss = loss_c+loss_src_align+loss_tgt_align
            loss.backward()            
            self.step_optimizer()
            ### ********************************************************************************


            if batch_idx > 500:
                return batch_idx        
            iteration, update = self.get_display_flag(batch_idx)
            if update:
                string1 = f' [Epo->{self.epoch+1}/{self.max_epoch}--Bat->{batch_idx+1}/{len(self.datasets)}] ({100.* iteration/self.max_iter:2.2f}%): '                
                print(f"Train {self.record_name}",
                      string1,
                      f"cls: {loss_c:.3f} src: {loss_src_align:.3f} tgt: {loss_tgt_align:.3f}")
                
                if self.write_logs:
                    df_dict = self.log_update_train_BCDA(loss_c, loss_src_align, loss_tgt_align, iteration)
                    if self.log_txt:
                        data_to_append = pd.DataFrame(df_dict, columns=df_dict.keys() if iteration==1 else None, index=[iteration])
                        data_to_append.to_csv(self.record_train_csv, index_label="iteration", mode='a', 
                                            header=True if (iteration==1 or not exists(self.record_train_csv)) else False) 

        self.epoch += 1
        return iteration, time.time()-start_time
    
    def train_M3SDA(self):
        start_time = time.time()
        self.model.train()
        for batch_idx, (features,labels) in enumerate(self.datasets):     
            features = [Variable(feature).cuda() for feature in features]
            labels = [Variable(label.long()).cuda() for label in labels]  
            ### ********************************************************************************
            ### ***************************    procedure 1    **********************************
            ###                             train G and (C1, C2)
            self.reset_grad()            
            feats, outputs = self.forward_all(features)
            loss_c, loss_msda = self.loss_all_domain(feats, outputs, labels)
            loss_s = [sum(loss_ci) for loss_ci in loss_c]
            loss = sum(loss_s)+loss_msda
            loss.backward()
            
            self.step_optimizer()
            ### ********************************************************************************


            ### ********************************************************************************
            ### ***************************    procedure 2    **********************************
            ###          fix G, and train (C1, C2) to diverse between Cs with L1-distance
            self.reset_grad()     
            feats, outputs = self.forward_all(features)
            loss_c, loss_msda = self.loss_all_domain(feats, outputs, labels)
            loss_s = [sum(loss_ci) for loss_ci in loss_c]
            loss = sum(loss_s)+loss_msda
  
            outputs = self.forward(features[-1])[1]
            loss_dis = self.discrepancy(outputs)
            loss = loss - loss_dis
            loss.backward()
            for opt in self.opt_c:
                opt.step()
            ### ********************************************************************************

            loss_diss=[loss_dis]
            ### ********************************************************************************
            ### ***************************    procedure 3    **********************************
            ### fix (C1, C2), and train G to minimize discrepency between preds of Cs with L1-distance 
            self.reset_grad()
            for _ in range(self.num_generator_update):
                outputs = self.forward(features[-1])[1]
                loss_dis = self.discrepancy(outputs)
                loss_dis.backward()
                self.opt_gs.step()
                self.reset_grad()
                loss_diss.append(loss_dis)
            ### ********************************************************************************    
            if batch_idx > 500:
                return batch_idx        
            iteration, update = self.get_display_flag(batch_idx)
            if update:
                string1 = f' [Epo->{self.epoch+1}/{self.max_epoch}--Bat->{batch_idx+1}/{len(self.datasets)}] ({100.* iteration/self.max_iter:2.2f}%): '
                string2 = '  '.join(f'Loss{num+1}-->{loss.data:.4e}' for num,loss in enumerate(loss_s)) \
                            + f'  Dis-->{loss_msda.data:.4e}'
                print(f"Train {self.record_name+string1+string2}")
                
                if self.write_logs:
                    df_dict = self.log_update_train(loss_s, loss_msda, loss_diss, iteration)
                    if self.log_txt:
                        data_to_append = pd.DataFrame(df_dict, columns=df_dict.keys() if iteration==1 else None, index=[iteration])
                        data_to_append.to_csv(self.record_train_csv, index_label="iteration", mode='a', 
                                            header=True if (iteration==1 or not exists(self.record_train_csv)) else False) 

        self.epoch += 1
        return iteration, time.time()-start_time
    
    def train_MFSAN(self):
        start_time = time.time()
        self.model.train()
        for batch_idx, (features,labels) in enumerate(self.datasets):     
            features = [Variable(feature).cuda() for feature in features]
            labels = [Variable(label.long()).cuda() for label in labels] 

            iteration, update = self.get_display_flag(batch_idx)
            gamma = 2 / (1 + exp(-10*iteration/self.max_iter)) - 1
            ### ********************************************************************************
            if update:
                title = f' [Epo-{self.epoch+1}/{self.max_epoch}--Bat-{batch_idx+1}/{len(self.datasets)}] ({100.* iteration/self.max_iter:2.2f}%):'
                print(f"Train {self.record_name+title}")
            
            fea_tgt, label_tgt = features.pop(),labels.pop()
            for num, (fea_src, label_src) in enumerate(zip(features,labels)):
                self.reset_grad()
                fea_src, pred_src, fea_tgts, pred_tgts = self.forward([fea_src,fea_tgt], idx=num, out_preds=True)
                loss_cls, loss_msda, loss_dis = self.loss_single_domain(fea_src, fea_tgts, pred_src, label_src, pred_tgts)

                loss = loss_cls + gamma * (loss_msda + loss_dis)           
                loss.backward()            
                self.step_optimizer()
            ### ********************************************************************************            
                if batch_idx > 500:
                    return batch_idx
                
                if update:
                    df_dict = {f"loss_cls_{num}": loss_cls.item(), 
                                f"loss_msda_{num}": loss_msda.item(), 
                                f"loss_dis_{num}": loss_dis.item()}
                    string3 = '  '.join(f'{key}-->{value:.4e}' for key,value in df_dict.items())
                    print(f"\t optimization for source {num+1}: {string3}")
                    if self.write_logs: 
                        update_callback(self.writer, iteration, df_dict)
                        if self.log_txt:
                            data_to_append = pd.DataFrame(df_dict, columns=df_dict.keys() if iteration==1 else None, index=[iteration])
                            data_to_append.to_csv(self.record_train_csv, index_label="iteration", mode='a', 
                                                header=True if (iteration==1 or not exists(self.record_train_csv)) else False) 

        self.epoch += 1
        return iteration, time.time()-start_time
    
    def train(self):
        if self.model_type=="M3SDA":
            return self.train_M3SDA()
        elif self.model_type=="MFSAN":
            return self.train_MFSAN()
        elif self.model_type=="BCDA":
            return self.train_BCDA()
        
    def validate(self, patience=10, stop_criterion=0.001):
        start_time = time.time()
        self.model.eval()
        size, iteration, acc, df_dict = 0, 0, 0, {}
        corrects = list(torch.zeros(self.num_source))
        for batch_idx, (features,labels) in enumerate(self.dataset_val):  
            # test on the target domain which placed in the last, source performance is not important
            features = [feature.cuda()  for feature in features]
            labels   = [label.long().cuda() for label in labels[:-1]]       

            outputs = self.forward_all(features)[1]

            size += (labels[0].shape[0])     
            corrects = torch.stack([correct+get_correct(output,label) for correct,output,label in zip(corrects,outputs,labels)])
            accs = 100.*corrects/size
            acc = sum(accs)/self.num_source
            iteration, update = self.get_display_flag(batch_idx, "val")
            if update or batch_idx==len(self.dataset_val)-1:
                string1 = f' [Epo-{self.epoch}/{self.max_epoch}--Bat-{batch_idx+1}/{len(self.dataset_val)}] ({100.* iteration/self.max_iter_val:2.2f}%): Acc'
                string2 = '\t'.join(f'On {self.sources[num]}-->{Correct}/{size} ({Acc:2.2f}%)' for num,(Correct,Acc) in enumerate(zip(corrects,accs)))
                print(f"validation {self.record_name+string1+string2}\tmean-->{acc}")
                if self.write_logs:  
                    df_dict = self.log_update_eval(acc, accs, iteration)
        self.val_acc = acc
        refresh_best = False  
        if self.val_acc_best<self.val_acc:
            self.val_acc_best=self.val_acc
            refresh_best = True
        if self.val_acc/100>1.0-1.0e-5 or (self.enable_early_stop and earlystop(self.val_acc/100, patience=patience, eps=stop_criterion)):
            self.early_stopped=True
                         
        if self.save_model:
            state_dict = dict(epoch=self.epoch, 
                              seed=self.seed, 
                              early_stopped=self.early_stopped, 
                              test_acc_best=self.test_acc_best,
                              model=self.model.state_dict())
            if self.epoch % self.save_period == 1:
                torch.save(state_dict, join(self.ckpt_dir, f'model_epoch_{self.epoch}.ckpt'))
            torch.save(state_dict, join(self.ckpt_dir,'model_last.ckpt'))
            if refresh_best:
                self.best_model=self.epoch
                torch.save(state_dict, join(self.ckpt_dir,'model_val_best.ckpt'))
                
        if self.write_logs and self.log_txt:
            data_to_append = pd.DataFrame(df_dict, columns=df_dict.keys() if iteration==1 else None, index=[iteration])
            data_to_append.to_csv(self.record_val_csv, mode='a', index_label="iteration", 
                                    header=True if (iteration==1 or not exists(self.record_val_csv)) else False) 
        return time.time()-start_time


    def test(self, patience=10, stop_criterion=0.001):
        start_time = time.time()
        self.model.eval()
        test_loss, correct, size, iteration, acc, df_dict = 0, 0, 0, 0, 0, {}
        corrects = list(torch.zeros(self.num_classifier))
        for batch_idx, (feature,label) in enumerate(self.dataset_test):  
            # test on the target domain which placed in the last, source performance is not important
            feature = Variable(feature[-1]).cuda()
            label = Variable(label[-1].long()).cuda()        

            outputs = self.forward(feature)

            size += (label.shape[0])     
            corrects = torch.stack([correct+get_correct(output,label) for correct,output in zip(corrects,outputs)])
            accs = 100.*corrects/size
            if self.ensemble_schema=='average':
                output = sum(list(outputs))/len(outputs)
            else:
                if sum(corrects)==0:
                    output = sum(list(outputs))/len(outputs)
                else:
                    output = sum([float(correct)*output for correct,output in zip(corrects, outputs)])/float(sum(corrects))
            test_loss += self.cls_crit(output, label)
            correct += get_correct(output,label)
            acc  = (100.*correct/size)
            test_loss = test_loss/ size

            iteration, update = self.get_display_flag(batch_idx, "test")
            if update:
                string1 = f' [Epo-{self.epoch}/{self.max_epoch}--Bat-{batch_idx+1}/{len(self.dataset_test)}] ({100.* iteration/self.max_iter_test:2.2f}%): '
                string2 = '  '.join(f'Acc{num+1}-->{Correct}/{size} ({Acc:2.2f}%)' for num,(Correct,Acc) in enumerate(zip(corrects,accs)))\
                         +f'  Acc->{correct}/{size} ({acc:.2f}%)'
                print(f"Test {self.record_name+string1+string2}")
                if self.write_logs:  
                    df_dict = self.log_update_eval(acc, accs, iteration, test_loss)
        print(f"Test {self.record_name}({100.* iteration/self.max_iter_test:2.2f}%) [Epo--{self.epoch}/{self.max_epoch}]: Acc-->{round(acc.item() ,2)}")
        self.test_acc = acc 
        refresh_best = False  
        if self.test_acc_best<self.test_acc:
            self.test_acc_best=self.test_acc
            refresh_best = True
        if self.test_acc/100>1.0-1.0e-5 or (self.enable_early_stop and earlystop(self.test_acc/100, patience=patience, eps=stop_criterion)):
            self.early_stopped=True
                         
        if self.save_model:
            state_dict = dict(epoch=self.epoch, 
                              seed=self.seed, 
                              early_stopped=self.early_stopped, 
                              test_acc_best=self.test_acc_best,
                              model=self.model.state_dict())
            if self.epoch % self.save_period == 1:
                torch.save(state_dict, join(self.ckpt_dir, f'model_epoch_{self.epoch}.ckpt'))
            torch.save(state_dict, join(self.ckpt_dir,'model_last.ckpt'))
            if refresh_best:
                self.best_model=self.epoch
                torch.save(state_dict, join(self.ckpt_dir,'model_best.ckpt'))
                
        if self.write_logs and self.log_txt:
            data_to_append = pd.DataFrame(df_dict, columns=df_dict.keys() if iteration==1 else None, index=[iteration])
            data_to_append.to_csv(self.record_test_csv, mode='a', index_label="iteration", 
                                    header=True if (iteration==1 or not exists(self.record_test_csv)) else False) 
        return time.time()-start_time

    def fit(self, patience=10, stop_criterion=0.001):
        if self.enable_early_stop:
            earlystop.count=0
        print(f"training {self.record_name} ...\n")
        Iter, Train_time, Test_time = [], [], []
        while self.epoch < self.max_epoch:
            iteration, train_time_consumption = self.train()
            test_time_consumption = self.test(patience=patience, stop_criterion=stop_criterion)
            Iter.append(iteration)
            Train_time.append(train_time_consumption)
            Test_time.append(test_time_consumption)
            if self.early_stopped:
                print(f"training {self.record_name} earlystopped at epoch {self.epoch}\n")
                self.early_stopped=True
                break
        if self.write_logs:            
            self.writer.close()
            self.events_to_mat()
        
        return Iter, Train_time, Test_time