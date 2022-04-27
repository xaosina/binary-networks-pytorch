from torch import optim as opt
import torch.optim.lr_scheduler as sch

OPT = {"SGD": lambda params_dict,
       lr=1e-3,
       momentum=0.9,
       weight_decay=0,
       betas=(0.9, 0.999),
       eps=1e-08: opt.SGD(params_dict,
                          lr=lr,
                          momentum=momentum,
                          weight_decay=weight_decay),
       "ADAM": lambda params_dict,
       lr=1e-3,
       momentum=0.9,
       weight_decay=0,
       betas=(0.9, 0.999),
       eps=1e-08: opt.Adam(params_dict,
                           lr=lr,
                           weight_decay=weight_decay,
                           betas=betas,
                           eps=eps),
       }

SCH = {
    "multistep": lambda optimizer,
    milestones=[40, 70, 80, 100, 110],
    gamma=0.1: sch.MultiStepLR(optimizer,
                               milestones=milestones, 
                               gamma=gamma),
    "cosine": lambda optimizer, epochs,
    eta_min=0,
    last_epoch=-1: sch.CosineAnnealingLR(optimizer, epochs, eta_min=eta_min, last_epoch=last_epoch),
    None: lambda *args, **kwargs: None
}
