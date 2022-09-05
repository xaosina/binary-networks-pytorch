import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.init as init
from models.common import LearnableBias
import torch.nn.functional as F

class BinarySoftActivation(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        output = (input == input.max(dim=1, keepdim=True)
                [0]).view_as(input).type_as(input)
        if not (output.sum(1) == 1).all():
            # print((output.sum(1) > 1).sum())
            print("More than one expert selected!!!!", input[output.sum(1) != 1])
            output = torch.zeros_like(output).scatter_(1, output.argmax(dim=-1).unsqueeze(1), 1.)
            assert (output.sum(1) == 1).all(), "Seriously??"
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_input


class TopKSampler(torch.autograd.Function):

    def __init__(self, k):
        super(TopKSampler, self).__init__()
        self.k = k
    
    @staticmethod
    def forward(ctx, input, k):
        # ctx.save_for_backward(input)
        output = input.multinomial(num_samples=k, replacement=True)
        output = torch.zeros_like(input).scatter_(1, output, 1.)
        output /= output.sum(1)[:, None]
        assert ((output > 0).sum(1) <= k).all(), output[(output > 0).sum(1) > k]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


