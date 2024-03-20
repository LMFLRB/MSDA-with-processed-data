from torch.nn.functional import softmax,log_softmax,nll_loss, cross_entropy
from torch import Tensor, stack

def l1_loss(source, target):
    return (softmax(source, dim=1) - softmax(target, dim=1)).abs().mean(-1).mean(-1)

def l1_loss_ms(targets):
    # target_num <--> [:]/target_num
    n_domain = len(targets)
    Targets = stack(targets) if type(targets) == list else targets
    Targets = Targets.unsqueeze(0).repeat(n_domain,1,1,1)
    discrepence = l1_loss(Targets, Targets.transpose(0,1)).sum()/(n_domain*(n_domain-1))
    return discrepence

def nll_cls_loss(pred_src, label_src):
    dense=lambda output: output.data.max(1)[1]
    return nll_loss(log_softmax(pred_src, dim=1), dense(label_src) if (len(label_src.size())>1 and label_src.size(1)>1) else label_src)

def ce_cls_loss(pred_src, label_src):
    return cross_entropy(pred_src, label_src)