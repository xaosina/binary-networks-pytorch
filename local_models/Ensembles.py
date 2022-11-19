import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.init as init
from models.common import LearnableBias
import torch.nn.functional as F

from .selectors.samplers import BinarySoftActivation, TopKSampler

class ConcatHead(nn.Module):
    def __init__(self, model, num_samples=1):
        super(ConcatHead, self).__init__()
        self.model = model
        self.model.eval()
        self.num_samples = num_samples
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(model.fc.in_features * num_samples, model.fc.out_features)

    def forward(self, input):
        assert self.model.training == False
        out = []
        for _ in range(self.num_samples):
            out += [self.model(input, get_embeddings=True)]
        out = self.fc(torch.cat(out, 1))
        return {"preds": out}

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self.fc.train(mode)
        return self

    def eval(self):
        return self.train(False)

class SumLogit(nn.Module):
    def __init__(self, model, num_samples=1):
        super(SumLogit, self).__init__()
        self.model = model
        self.model.eval()
        self.num_samples = num_samples
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input):
        assert self.model.training == False
        out = 0
        for _ in range(self.num_samples):
            out += self.model(input)
        out /= self.num_samples
        return {"preds": out}

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

class CustomConcat(ConcatHead):
    def __init__(self, model):
        super(CustomConcat, self).__init__(model, 4)

    def forward(self, input):
        assert self.model.training == False
        
        self.model.sampling["k"] = None
        out = [self.model(input, get_embeddings=True)]
        out += [self.model(input.flip((3,)), get_embeddings=True)]

        self.model.sampling["k"] = 1
        out += [self.model(input, get_embeddings=True)]
        out += [self.model(input.flip((3,)), get_embeddings=True)]
        
        out = self.fc(torch.cat(out, 1))
        return {"preds": out}

class CustomSum(SumLogit):
    def __init__(self, model):
        super(CustomSum, self).__init__(model, 4)

    def forward(self, input):
        assert self.model.training == False
        
        self.model.sampling["k"] = None

        out = self.model(input)["preds"]
        out += self.model(input.flip((3,)))["preds"]

        self.model.sampling["k"] = 1
        out += self.model(input)["preds"]
        out += self.model(input.flip((3,)))["preds"]
    
        out /= 4
        return {"preds": out}
