import torch.optim as optim
import torch.nn.functional as F

LOSS_FNS = {
    "mse": F.mse_loss,
    "l1": F.l1_loss,
    "smooth_l1": F.smooth_l1_loss,
    "binary_cross_entropy": F.binary_cross_entropy,
    "cross_entropy": F.cross_entropy,
    "kl_div": F.kl_div,
    "huber": F.huber_loss,
    "binary_cross_entropy_with_logits": F.binary_cross_entropy_with_logits,
}

OPTIMIZER_FNS = {
    "adam": optim.Adam,
    "adam_w": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "lbfgs": optim.LBFGS,
}

def get_loss_fn(loss_name):
    return LOSS_FNS[loss_name]

def get_optimizer_fn(optimizer_name):
    return OPTIMIZER_FNS[optimizer_name]