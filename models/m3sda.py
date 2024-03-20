import torch.nn as nn
from torch import stack
from .base import *
class M3SDA(nn.Module):
    def __init__(self,
                    num_class:int=10,
                    num_classifier:int=2, 
                    patch_size:int=32, 
                    feat_hiddens:list=[64,64,128],
                    cls_hiddens:list=None,
                    lambd:float=1.0,   
                    is_image:bool=True,
                    use_resnet:bool=False,
                    resnet_type:str='ResNet50',
                    dim_input:int=2048,
                    train_resnet:bool=False,
                    with_processed_data:bool=False,
                    feature_type:str="linear",
                    **kwargs
                    ):
        super(M3SDA, self).__init__()
        self.num_class=num_class
        self.num_classifier=num_classifier
        self.patch_size=patch_size
        self.lambd=lambd
        
        if with_processed_data:
            if feature_type=="linear":
                num_latent = 256
                self.sharedNets = FeatureFC([dim_input]+feat_hiddens+[num_latent])
            else:
                feat_hiddens=[512,256,128]
                self.sharedNets = FeatureConv2d(512 if int(resnet_type[6:]) in [18,34] else 2048,
                                                feat_hiddens,
                                                [3,3,3],[2,2,2],[1,1,1],
                                                patchsize=7)
                num_latent = feat_hiddens[-1]*self.sharedNets.patchsize**2
        else:
            if is_image:
                if use_resnet:
                    if train_resnet and dim_input==224:
                        self.sharedNets = nn.Sequential(globals()[resnet_type], 
                                                    nn.AvgPool2d(7, stride=1), 
                                                    Flatten())
                        num_latent = 2048
                    else:        
                        self.sharedPre=globals()[resnet_type.lower()](
                            weights=f"ResNet{int(resnet_type[6:])}_Weights.DEFAULT",
                            feature_only=True,
                        )
                        # freeze the pretrained parameters to no backpropogation
                        for param in self.sharedPre.parameters():
                            param.requires_grad = False
                        feat_hiddens=[512,256,128]
                        self.sharedNets = FeatureConv2d(2048,feat_hiddens,[3,3,3],[2,2,2],[1,1,1],patchsize=7)
                        num_latent = feat_hiddens[-1]*self.sharedNets.patchsize**2
                else:            
                    # saving memory configs:
                    self.sharedNets = FeatureConv2d(3,feat_hiddens,[5,3,3],[2,1,1],[1,1,1],patchsize=patch_size)
                    num_latent = feat_hiddens[-1]*self.sharedNets.patchsize**2
                    # saving memory configs
                    
                    # ## configs from m3sda    ******************************************************
                    # cov2d=FeatureConv2d(3,feat_hiddens,patchsize=patch_size)
                    # num_latent = feat_hiddens[-1]*cov2d.patchsize**2
                    # self.sharedNets = nn.Sequential(cov2d, 
                    #                                FeatureFC([num_latent,3092], use_dropout=True),
                    #                                FeatureFC([3092,2048]))
                    # num_latent = 2048
                    # ## configs from m3sda    ******************************************************
            else: # pre-extracted features by ResNet or other networks, the feture dimmension dim_input needed
                num_latent = 256
                self.sharedNets = FeatureFC([dim_input,num_latent,num_latent,num_latent])
            
        self.clsFCs = Predictor(num_latent,num_class,cls_hiddens) if num_classifier==1 else \
                 nn.ModuleList([Predictor(num_latent,num_class,cls_hiddens) for _ in range(num_classifier)])
        self.num_latent = num_latent
    
    def forward_feat(self, src, *kwargs):
        return self.sharedNets(getattr(self, 'sharedPre')(src) if hasattr(self, 'sharedPre') else src)
    
    def forward_pred(self, feat, reverse=False, **kwrags):
        return self.clsFCs(feat, reverse=reverse) if self.num_classifier==1 \
                else stack([cls(feat, reverse=reverse) for cls in self.clsFCs])

    def forward(self, src, **kwargs):
        feat = self.forward_feat(src)
        preds= self.forward_pred(feat, **kwargs)        
        return (feat, preds) if self.training else preds
    
    def forward_all(self, srcs, **kwargs):
        # preds: [n_src,n_cls,...] to [n_cls,n_src,...]?
        # feats = [self.forward_feat(src) for src in srcs]
        # preds = stack([self.forward_pred(feat, **kwargs) for feat in feats]).transpose(0,1)  
        feats, preds = [], []
        for src in srcs:
            feat, pred = self.forward(src, **kwargs)
            feats.append(feat)
            preds.append(pred)

        return feats, stack(preds).transpose(0,1)  
        
    def cls_loss_all_domain(self, outputs, labels, crit):
        return [crit(output, label) for output, label in zip(outputs,labels)]
    
    def loss_all_domain(self, feats, outputs, labels, cls_crit, msda_crit, msda_params):
        loss_c = [self.cls_loss_all_domain(output, labels, cls_crit) for output in outputs]

        weight = msda_params.pop('msda_weight')
        loss_msda = weight*msda_crit(*feats, **msda_params)

        return loss_c, loss_msda
    