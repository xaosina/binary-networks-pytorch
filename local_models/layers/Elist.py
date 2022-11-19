import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.common_types import _size_2_t
import torch.fx

import math
import numpy as np
from functools import partial
from typing import Union
import warnings

from bnn import BConfig, bconfig
from bnn.layers.helpers import copy_paramters

@torch.fx.wrap
def expert_binary_conv2d(*args, **kwargs):
    return F.conv2d(*args, **kwargs)

# @torch.fx.wrap
# def expert_conv2d(*args, **kwargs):
#     return F.conv2d(*args, **kwargs)

def EconvList(modules, gate_x, x):
    x = x.clone()
    B, C, H, W = x.shape
    if isinstance(x, torch.fx.proxy.Proxy):
        warnings.warn("Proxy round. Loose your hope for have adequate computations. Also batch size HAVE TO BE 1. OR ELSE.")
        B = 1
    base_weight = torch.stack([m.weight_pre_process(m.weight) for m in modules])
    for i in range(len(modules)):
        x[gate_x.argmax(1) == i] = modules[i].activation_pre_process(x[gate_x.argmax(1) == i])
    weight = torch.matmul(
        gate_x,
        base_weight.view(len(modules), -1)
    ).view(B * modules[0].out_channels, modules[0].in_channels // modules[0].groups, modules[0].kernel_size[0], modules[0].kernel_size[1])
    x = x.view(1, B * C, H, W)

    conv = expert_binary_conv2d
    out = conv(
        x, weight, None, stride=modules[0].stride, padding=modules[0].padding,
        dilation=modules[0].dilation, groups=modules[0].groups * B
    )
    out = out.permute([1, 0, 2, 3]).view(
        B, modules[0].out_channels, out.shape[-2], out.shape[-1])

    for i in range(len(modules)):
        out[gate_x.argmax(1) == i] = modules[i].activation_post_process(out[gate_x.argmax(1) == i], x)
    
    return out

def EBnnList(modules, gate_x, x):
    x = x.clone()
    for i in range(len(modules)):
        x[gate_x.argmax(1) == i] = modules[i](x[gate_x.argmax(1) == i])
    return x

def EReluList(modules, gate_x, x):
    out = []
    for i in range(len(modules)):
        out += [modules[i](x)]
    final_output = torch.stack(out)
    final_output = torch.sum(gate_x.T[:, :, None, None, None] * final_output, 0)
    return final_output

def EScaleList(modules, gate_x, x):
    x = x.clone()
    for i in range(len(modules)):
        x[gate_x.argmax(1) == i] = modules[i] * (x[gate_x.argmax(1) == i])
    return x