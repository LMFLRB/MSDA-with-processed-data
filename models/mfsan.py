import torch.nn as nn
from copy import deepcopy as copy
from .base import *

class MFSAN(nn.Module):
    def __init__(self, 
                num_class=31, 
                num_source=2, 
                patch_size=224, 
                feat_hiddens:list=[64,64,128],   
                is_image:bool=True,
                use_resnet:bool=False,
                resnet_type:str='ResNet50',
                dim_input:int=2048,
                **kwargs
                ):
        super(MFSAN, self).__init__()
        self.num_class = num_class
        self.num_source = num_source
        self.use_resnet = use_resnet
        if is_image:
            if self.use_resnet:
                self.sharedNets = globals()[resnet_type]
                # for param in self.sharedNets.parameters():
                #     param.requires_grad = False
                num_latent = 256
                self.sonNets = nn.ModuleList([nn.Sequential(
                                            ADDneck([2048]+[num_latent]*3),
                                            nn.AvgPool2d(7, stride=1),
                                            Flatten()) for _ in range(num_source)])
            else:
                self.sharedNets = ADDneck([3]+feat_hiddens,
                                        kernels=[5,3,3],
                                        paddings=[1,1,1],
                                        strides=[2,2,2],
                                        patchsize=patch_size)
                hiddens=[feat_hiddens[-1]]+feat_hiddens
                self.sonNets = nn.ModuleList([nn.Sequential(
                                            ADDneck(hiddens, patchsize=self.sharedNets.patchsize),
                                            Flatten(),
                                            ) for _ in range(num_source)])     
                num_latent = hiddens[-1]*self.sonNets[0][0].patchsize**2  
        else:
            num_latent = 256
            self.sharedNets = FeatureFC([dim_input]+[num_latent]*3)
            self.sonNets = nn.ModuleList([FeatureFC([num_latent,num_latent]) for _ in range(num_source)])     
        self.clsFCs = nn.ModuleList([Predictor(num_latent,num_class) for _ in range(num_source)])


    def forward(self, data_srcs,  idx=0, out_preds=False, **kwrags):
        if self.training == True:
            fea_src = self.sonNets[idx](self.sharedNets(data_srcs[0]))
            pred_src = self.clsFCs[idx](fea_src)
            fea_tgt_ = self.sharedNets(data_srcs[1])
            fea_tgts = [sonNet(fea_tgt_) for sonNet in self.sonNets]

            # mmd_losses = [losses.mmd(fea_src, fea_tgt) for fea_src,fea_tgt in zip(fea_srcs,fea_tgts)]
            # cls_losses= [losses.cls_loss(pred_src, label_src) for pred_src,label_src in zip(pred_srcs,label_srcs)]
            # l1_losses = losses.l1_loss_ms(fea_tgts)
            if out_preds:
                pred_tgts = [clsFC(fea_tgt) for clsFC, fea_tgt in zip(self.clsFCs, fea_tgts)]
                return fea_src, pred_src, fea_tgts, pred_tgts
            else:
                return fea_src, pred_src, fea_tgts
        else:
            fea_src = self.sharedNets(data_srcs)
            pred_srcs = [clsFC(sonNet(fea_src)) for clsFC,sonNet in zip(self.clsFCs,self.sonNets)]
            return pred_srcs
    
    def forward_all(self, datas, out_preds=False, **kwrags):        
        data_srcs = copy(datas)
        data_tgt = data_srcs.pop()
        if self.training == True:
            fea_srcs = [sonNet(self.sharedNets(data_src)) for data_src,sonNet in zip(data_srcs,self.sonNets)]
            pred_srcs = [clsFC(fea_src) for clsFC, fea_src in zip(self.clsFCs, fea_srcs)]
            fea_tgt_ = self.sharedNets(data_tgt)
            fea_tgts = [sonNet(fea_tgt_) for sonNet in self.sonNets]
            
            if out_preds:
                pred_tgts = [clsFC(fea_tgt) for clsFC, fea_tgt in zip(self.clsFCs, fea_tgts)]
                return fea_srcs, pred_srcs, fea_tgts, pred_tgts
            else:
                return fea_srcs, pred_srcs, fea_tgts

        else:
            fea = self.sharedNets(data_srcs)
            pred_srcs = [clsFC(sonNet(fea)) for clsFC,sonNet in zip(self.clsFCs,self.sonNets)]
            return tuple(pred_srcs)
        
    def loss_single_domain(self, 
                           fea_src, fea_tgts, 
                           pred_src, pred_tgts,
                           label_src, 
                           cls_crit, 
                           dis_crit,
                           msda_crit, 
                           msda_params):
        
        weight = msda_params.pop('msda_weight')

        loss_cls = cls_crit(pred_src, label_src)
        loss_dis = dis_crit(pred_tgts)
        loss_msda =  weight*msda_crit(*(fea_tgts+[fea_src]), **msda_params)

        return loss_cls, loss_msda, loss_dis