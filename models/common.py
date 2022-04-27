import torch
import torch.nn as nn


class LearnableBias(torch.nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(
            torch.zeros(1, out_chn, 1, 1), requires_grad=True
        )

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class LearnableAlpha(torch.nn.Module):
    def __init__(self, out_chn):
        super(LearnableAlpha, self).__init__()
        self.alpha = nn.Parameter(
            torch.ones(1, out_chn,1,1), requires_grad=True
        )

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha)
        #print(alpha.flatten())
        out = x * alpha.expand_as(x)
        #print(self.alpha.flatten())
        return out