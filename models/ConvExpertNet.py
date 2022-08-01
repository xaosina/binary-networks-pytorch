import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.init as init
from models.common import LearnableBias
import torch.nn.functional as F

from .layers.ebconv import EBConv2d


def conv3x3(
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        num_experts: int = 4,
        ebconv_activation=torch.softmax,
        use_only_first: bool = False,
        use_se: bool = True) -> nn.Module:
    """3x3 convolution with padding"""
    if num_experts > 1:
        return EBConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
            num_experts=num_experts,
            activation=ebconv_activation,
            use_only_first=use_only_first,
            use_se=use_se)
    else:
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation)


def conv1x1(
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        num_experts: int = 4,
        ebconv_activation=torch.softmax,
        use_only_first: bool = False,
        use_se: bool = True) -> nn.Module:
    """1x1 convolution"""
    if num_experts > 1:
        return EBConv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            bias=False,
            num_experts=num_experts,
            activation=ebconv_activation,
            use_only_first=use_only_first,
            use_se=use_se)
    else:
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            bias=False)

class RPReLU(nn.Module):
    def __init__(self, channels, bias=False):
        super(RPReLU, self).__init__()
        self.prelu = nn.PReLU(channels)
        if bias:
            self.inp_bias = LearnableBias(channels)          
            self.out_bias = LearnableBias(channels)
        self.bias = bias

    def forward(self, x):
        if self.bias: 
            x = self.inp_bias(x)
        x = self.prelu(x)
        if self.bias:
            x = self.out_bias(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, num_bases, inplanes, planes, stride=1, downsample=None, rprelu=False, use_only_first=False
    ):
        super(BasicBlock, self).__init__()
        self.rprelu = rprelu
        self.use_only_first = use_only_first
        print("RPReLU ", rprelu, num_bases) 
        self.num_bases = num_bases
        self.conv1 = conv3x3(inplanes, planes, stride, num_experts=num_bases, use_only_first=use_only_first)       
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = RPReLU(planes, bias=rprelu)
        self.conv2 = conv3x3(planes, planes, num_experts=num_bases, use_only_first=use_only_first)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = RPReLU(planes, bias=rprelu)
        self.downsample = downsample
        self.scale = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.activation = torch.softmax
        # self.fc = nn.Linear(inplanes, num_bases)

    def forward(self, x):

        final_output = None
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out += residual

        out_new = self.conv2(out)
        out_new = self.bn2(out_new)
        out_new = self.relu2(out_new)
        out_new += out
        
        final_output = self.scale * out_new 
        return final_output



class downsample_layer(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, bias=False, use_only_first=False):
        super(downsample_layer, self).__init__()
        self.conv = conv1x1(
            inplanes, 
            planes, 
            stride=stride, 
            use_only_first=use_only_first
        )
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return x



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, rprelu=False, use_only_first=False):
        self.inplanes = 64
        self.num_bases = 4
        self.use_only_first = use_only_first
        self.rprelu = rprelu
        super(ResNet, self).__init__()
        self.first_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True) #don't quantize the first layer
        self.bn1 = nn.BatchNorm2d(64) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = downsample_layer(self.inplanes, planes * block.expansion, 
                          kernel_size=1, stride=stride, bias=False, use_only_first=self.use_only_first)

        layers = []
        layers.append(block(
            self.num_bases, 
            self.inplanes, 
            planes, 
            stride, 
            downsample, 
            rprelu=self.rprelu,
            use_only_first=self.use_only_first
        ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.num_bases, 
                self.inplanes, 
                planes, 
                rprelu=self.rprelu, 
                use_only_first=self.use_only_first
            ))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)
        x5 = self.fc(x4)

        return {"preds": x5}


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        load_dict = torch.load('./full_precision_weights/model_best.pth.tar')['state_dict']
        model_dict = model.state_dict()
        model_keys = model_dict.keys()
        for name, param in load_dict.items():
            if name.replace('module.', '') in model_keys:
                model_dict[name.replace('module.', '')] = param    
        model.load_state_dict(model_dict)  
    return model


def resnet34(bitW, bitA, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], bitW, bitA, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(bitW, bitA, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], bitW, bitA, **kwargs)
    return model
