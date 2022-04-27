import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import LearnableBias, LearnableAlpha


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, stride=5)
        self.move = LearnableBias(8)
        # self.move1_a = LearnableBias(128)
        # self.conv2 = nn.Conv2d(128, 64, 3, 1)
        # self.move2_c = LearnableBias(128)
        # self.move2_a = LearnableBias(64)

        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32, 10)
        # self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # x = self.move1_c(x)
        x = self.conv1(x)
        x = self.move(x)
        # x = self.move1_a(x)
        x = F.relu(x)
        # x = self.move2_c(x)
        # x = self.conv2(x)
        # x = self.move2_a(x)
        # x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # print(self.move2_a.bias)
        output = F.log_softmax(x, dim=1)
        # print(self.conv1.weight)
        return {"preds": output}



class SignActivation(torch.autograd.Function):
    r"""Applies the sign function element-wise
    :math:`\text{sgn(x)} = \begin{cases} -1 & \text{if } x < 0, \\ 1 & \text{if} x >0  \end{cases}`
    the gradients of which are computed using a STE, namely using :math:`\text{hardtanh(x)}`.
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> input = torch.randn(3)
        >>> output = SignActivation.apply(input)
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_input

class MnistNetRandom(nn.Module):
    def __init__(self):
        super(MnistNetRandom, self).__init__()
        f = 256
        self.conv1 = nn.Conv2d(1, f, 3, stride=1, bias=False)
        self.move1 = LearnableBias(f)
        #self.alpha1 = LearnableAlpha(f)
        self.conv2 = nn.Conv2d(f, f, 3, stride=5, bias=False)
        self.move2 = LearnableBias(f)
        #self.alpha2 = LearnableAlpha(f)
        self.fc = nn.Linear(256, 10, bias=False)
       

    def forward(self, x):
        x = SignActivation.apply(x)
        x = self.conv1(x)
        x = self.move1(x)

        #x = self.alpha1(x)
        #print(self.alpha1.alpha.flatten())
        
        x = F.leaky_relu(x)
        x = SignActivation.apply(x)
        x = self.conv2(x)
        x = self.move2(x)
        #x = self.alpha2(x)
        
        x = F.max_pool2d(x, 5)
        x = F.leaky_relu(x)
        x = torch.flatten(x, 1)
        x = SignActivation.apply(x)
        x = self.fc(x)
        #x = self.alpha(x)
        output = F.log_softmax(x, dim=1)
        return {"preds": output}


def bin_fn(C):
    C.weight = nn.parameter.Parameter(torch.sign(C.weight), requires_grad=False)
    #C.weight.requries_grad = False
    return C

def replace_module_with(root_module, init_fn, skip_names=[]):
    new_dict = root_module._modules.copy()

    def apply(m):
        for name, child in m.named_children():
            if not name in skip_names:
                f = init_fn(child)
                setattr(m, name, f)
                new_dict[name] = f
            else:
                apply(child)

    apply(root_module)


def get_binary_mnist():
    model = MnistNetRandom()
    replace_module_with(model, bin_fn, skip_names=['move1','move2','alpha2','alpha1'])
    return model