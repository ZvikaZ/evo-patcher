import torch
import logging
from misc import get_device

logger = logging.getLogger(__name__)
device = get_device()


def tensorize(x):
    return torch.tensor(x).float().to(device)


def t_add(x, y):
    """x+y"""
    return torch.add(tensorize(x), tensorize(y))


def t_sub(x, y):
    """x-y"""
    return torch.subtract(tensorize(x), tensorize(y))


def t_mul(x, y):
    """x*y"""
    return torch.multiply(tensorize(x), tensorize(y))


def t_div(x, y):
    """protected division: if abs(y) > 0.001 return x/y else return 0"""
    # with torch.errstate(divide='ignore', invalid='ignore'):
    return torch.where(torch.abs(tensorize(y)) > 0.001, torch.divide(tensorize(x), tensorize(y)), 0.)


def t_sin(x):
    """sin(x)"""
    return torch.sin(tensorize(x))


def t_cos(x):
    """cos(x)"""
    return torch.cos(tensorize(x))


def t_atan2(x, y):
    """atan2(y,x)"""
    return torch.atan2(tensorize(x), tensorize(y))


def t_hypot(x, y):
    """hypot(x,y)"""
    try:
        return torch.hypot(tensorize(x), tensorize(y))
    except RuntimeError:
        print(x)
        print(y)
        print("ha")
        return 1


def t_iflte(x, y, z, w):
    """if x <= y return z else return w"""
    return torch.where(tensorize(x) <= 0, y, z)
