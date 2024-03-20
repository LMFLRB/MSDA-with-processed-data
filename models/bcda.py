import torch.nn as nn
from torch import stack, concat
from .base import *

class BCDA(nn.Module):
    name = 'BCDA'
    def __init__(self,
                    num_class:int=10,
                    num_source:int=2, 
                    patch_size:int=32, 
                    dim_input:int=2048,
                    feat_hiddens:list=[1024,512,512],
                    cls_hiddens:list=None,
                    lambd:float=1.0,  
                    resnet_type:str='ResNet50',
                    num_latent:int=256,
                    with_processed_data:bool=False,
                    feature_type:str="linear",
                    **kwargs
                    ):
        super(BCDA, self).__init__()
        self.num_class=num_class
        self.num_source=num_source
        self.patch_size=patch_size
        self.lambd=lambd
        self.num_latent = num_latent  
        if with_processed_data:
            if feature_type=="linear":
                self.sharedNets = nn.ModuleList([FeatureFC([dim_input]+feat_hiddens+[self.num_latent])
                                        for _ in range(num_source+1)])
            else:
                # feat_hiddens_=[512,256,128]
                self.sharedNets = nn.ModuleList([FeatureConv2d(2048,
                                                               feat_hiddens,
                                                               [3,3,3],
                                                               [2,2,2],
                                                               [1,1,1],
                                                               patchsize=7)
                                        for _ in range(num_source+1)])
                self.num_latent = feat_hiddens[-1]*self.sharedNets.patchsize**2

        else:
            self.sharedPre = globals()[resnet_type] #ResNet50 # ResNet101
            # freeze the pretrained parameters to no backpropogation
            for param in self.sharedPre.parameters():
                param.requires_grad = False
            # using Conv2d
            # self.sharedNets = FeatureConv2d(2048,feat_hiddens,[3,3,3],[2,2,2],[1,1,1])
            # self.num_latent = feat_hiddens[-1]*self.sharedNets.patchsize**2
            # using FC      
            self.sharedNets = nn.ModuleList([nn.Sequential(nn.AdaptiveMaxPool2d((1,1)),
                                        Flatten(start_dim = 1),
                                        FeatureFC([2048]+feat_hiddens+[self.num_latent]))
                                        for _ in range(num_source)])
        self.fusionNet = Attension(self.num_latent)
        self.clsFCs = Predictor(self.num_latent,self.num_class,cls_hiddens)
    
    def forward_feat(self, src, src_num:int=0, **kwargs):
        return self.sharedNets[src_num](self.sharedPre(src) if hasattr(self, 'sharedPre') else src)
    
    def forward_pred(self, feat, reverse=False, **kwrags):
        return self.clsFCs(feat, reverse=reverse)

    def forward(self, src, src_num:int=0, **kwargs):
        feat = self.forward_feat(src, src_num, **kwargs)
        pred = self.forward_pred(feat, **kwargs)        
        return (feat, pred) if self.training else [pred]
    
    def forward_all(self, srcs, **kwargs):
        # preds: [n_src,n_cls,...] to [n_cls,n_src,...]?
        # feats = [self.forward_feat(src) for src in srcs]
        # preds = stack([self.forward_pred(feat, **kwargs) for feat in feats]).transpose(0,1)  
        feats = [self.forward_feat(src, num, **kwargs) for num, src in enumerate(srcs)]
        preds = self.forward_pred(concat(feats[:-1],0), **kwargs)     
        feat_bc = self.fusionNet(stack(feats[:-1]))       
        return (feats[:-1], feats[-1], feat_bc), preds
    
    def loss_all_domain(self, feats, outputs, labels, cls_crit, msda_crit, msda_params):
        loss_c = cls_crit(outputs, concat(labels,0))
        weight = msda_params.pop('msda_weight')
        feats_src, feat_tgt, feat_bc =feats
        loss_src_bc = weight*msda_crit(*(feats_src+[feat_bc]), **msda_params)        
        loss_tgt_bc = weight*msda_crit(*([feat_tgt,feat_bc]), **msda_params)

        return loss_c, loss_src_bc, loss_tgt_bc
    
    def loss_src_align(self, feats_src, feat_bc, msda_crit, msda_params):
        weight = msda_params.pop('msda_weight')      
        loss_src_bc = weight*msda_crit(*(feats_src+[feat_bc]), **msda_params)
        return loss_src_bc
    
    def loss_tgt_align(self, feat_tgt, feat_bc, msda_crit, msda_params):
        weight = msda_params.pop('msda_weight')      
        loss_tgt_bc = weight*msda_crit(*([feat_tgt,feat_bc]), **msda_params)
        return loss_tgt_bc
    