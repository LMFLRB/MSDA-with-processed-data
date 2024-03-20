import torch
from numpy import ndarray
import warnings
from typing import Union

warnings.filterwarnings("ignore")

def atleast_epsilon(X, eps=1.0e-10):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return torch.where(X < eps, X.new_tensor(eps), X)

def p_dist_2(x, y):
    # x, y should be with the same flatten(1) dimensional
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    x_norm2 = torch.sum(x**2, -1).reshape((-1, 1))
    y_norm2 = torch.sum(y**2, -1).reshape((1, -1))
    dist = x_norm2 + y_norm2 - 2*torch.mm(x, y.t())
    return torch.where(dist<0.0, torch.zeros(1).to(dist.device), dist)

def calculate_gram_DG(domains:Union[list,tuple,torch.Tensor], 
                      sigmas: Union[torch.Tensor,ndarray,float]=None,
                      **kwargs)->torch.Tensor:
    dn, bn = len(domains), len(domains[0])
    PD = torch.zeros([dn,dn,bn,bn], device=domains[0].device)
    out_sigmas=[]
    for t in range(dn):
        for k in range(t+1):
            pd_tk = p_dist_2(domains[t], domains[k])
            if sigmas==None:
                # sigma = (0.15*pd_tk[np.triu_indices(len(pd_tk))].median())
                sigma = pd_tk.sum()/(bn**2-bn)
            else:
                sigma = sigmas[t,k] if type(sigmas) in [torch.Tensor,ndarray] else sigmas
            PD[t,k,...] = PD[k,t,...] = -pd_tk/sigma
            out_sigmas.append(sigma)
    
    return PD.exp(), out_sigmas   

def calculate_gram_TS(domains:Union[list,tuple,torch.Tensor], 
                      sigmas: Union[torch.Tensor,ndarray,float]=None,
                      **kwargs)->torch.Tensor:
    dn, bn = len(domains), len(domains[0])
    domain_expand = (domains if isinstance(domains, torch.Tensor) else torch.stack(domains)).flatten(2)
    domain_expand = domain_expand.unsqueeze(0).unsqueeze(2).repeat(dn,1,bn,1,1)
    domain_expand_T = domain_expand.permute(1,0,3,2,4)
    
    PD = ((domain_expand-domain_expand_T)**2).sum(-1)
    if sigmas==None:
        sigmas = (PD.sum([-1,-2])/(bn**2-bn)) # for mean estimation
        PD = PD/sigmas.unsqueeze(-1).unsqueeze(-1)
    else:
        PD = PD/sigmas.unsqueeze(-1).unsqueeze(-1)

    return (-PD).exp(), sigmas

def mhsic(samplesets:Union[list,tuple,torch.Tensor], 
               sigmas: Union[torch.Tensor,ndarray,float]=None, 
               **kwargs)->torch.Tensor:
    from numpy import arange, triu_indices
    dn, bn = len(samplesets), len(samplesets[0])
    # K = calculate_gram_TS(samplesets,sigmas,**kwargs)[0]
    K = calculate_gram_DG(samplesets,sigmas,**kwargs)[0]

    idx_diag = arange(dn)
    idx_triu = triu_indices(dn, 1)
    Diag = K[idx_diag, idx_diag, ...]
    Cross= K[idx_triu[0], idx_triu[1], ...]
    mhsic = Diag.prod(0).mean() + Diag.mean([-1,-2]).prod() - 2.*Cross.mean([-1,-2]).prod()
    return mhsic

def gcsd(*domains:Union[list,tuple,torch.Tensor], 
         sigmas: Union[torch.Tensor,ndarray,float]=None,
         **kwargs) -> torch.tensor:
    """
    math:
        domains: mini-batch of features for all domains with shape--> n \times dim
        gram_mat: pairwise gram matrix between different domains to form a 4-dim tensor K---> s \time s \times n \time s
        \begin{array}{c}
        {D_{GCS}} \approx  
        - \log \left( {\frac{1}{s}\sum\limits_{t = 1}^s {\frac{1}{n}\sum\limits_{j = 1}^n {\prod\limits_{k \ne t}^s {\frac{1}{n}\sum\limits_{i = 1}^n {{\kappa _\sigma }\left( {{\rm{x}}_j^t - {\rm{x}}_i^k} \right)} } } } } \right) 
        + \frac{1}{s}\sum\limits_{t = 1}^s {\log \left( {\frac{1}{n}\sum\limits_{j = 1}^n {{{\left( {\frac{1}{n}\sum\limits_{i = 1}^n {{\kappa _\sigma }\left( {{\rm{x}}_i^t - {\rm{x}}_j^t} \right)} } \right)}^{s - 1}}} } \right)} \\
        = - \log \left( {\left( {\frac{{K.{\rm{mean}}\left( { - 1} \right).{\rm{prod}}\left( 1 \right)}}{{K.{\rm{mean}}\left( { - 1} \right).{\rm{diag}}\left( {0,1} \right)}}} \right).{\rm{mean}}\left( {} \right)} \right) 
          + \frac{1}{s}\sum\limits_{t = 1}^s {\log \left( {\frac{1}{{{N^s}}}\sum\limits_{j = 1}^N {{{\left( {K.{\rm{mean}}\left( { - 1} \right).{\rm{diag}}\left( {0,1} \right)} \right)}^{s - 1}}} } \right)} 
        \end{array}
    """
    # domain gram_mat
    # gram_mat = calculate_gram_TS(domains, sigmas, **kwargs)[0]
    gram_mat = calculate_gram_DG(domains, sigmas, **kwargs)[0] # --> make s*s*n*n dimmension tensor K
    # GCSD-Inter-Domain discrepancy
    s = gram_mat.shape[0]
    tri_d = torch.arange(s)                       

    Gs=gram_mat.mean(-1)
    Gp=Gs[tri_d,tri_d,...]  # --> make diag K of the domain dimmension
    Gc=Gs.prod(1)/Gp

    cross_entropy=-atleast_epsilon(Gc,       ).mean(-1).mean().log()    
    power_entropy=-atleast_epsilon(Gp**(s-1)).mean(-1).log().mean()

    return (cross_entropy - power_entropy)/s

def gjrd(*domains:Union[list,tuple,torch.Tensor], 
         sigmas: Union[torch.Tensor,ndarray,float]=None,
         **kwargs) -> torch.tensor:
    # domain gram_mat
    # gram_mat = calculate_gram_TS(domains, sigmas, **kwargs)[0]
    gram_mat = calculate_gram_DG(domains, sigmas, **kwargs)[0]
    # GJRD-Inter-Domain discrepancy
    if 'order' or 'params' in kwargs:
        order = kwargs.get('params') if kwargs.get('order') is None else kwargs.get('order')
    else:
        order = 2
    dn = gram_mat.shape[0]

    if 'weights' in kwargs:
        B = torch.Tensor(kwargs['weights']).to(domains[0].device)
        if len(B)==2 and dn>2:
            B = torch.Tensor([B[0]/(dn-1)]*(dn-1)+[B[1]]).to(domains[0].device).view(-1,1,1)/B.sum()
    else:
        B = torch.ones([dn,1,1]).to(domains[0].device)
    
    tri_d = torch.arange(dn)

    Gp = gram_mat.mean(-1)
    G1 = ((B*Gp).sum(1))**(order-1)    
    G2 = (Gp[tri_d,tri_d,:])**(order-1)    

    cross_entropy=atleast_epsilon(G1).mean().log()
    power_entropy=(atleast_epsilon(G2).mean(-1).log()*B.squeeze()).sum()

    return (cross_entropy - power_entropy)/(1-order)