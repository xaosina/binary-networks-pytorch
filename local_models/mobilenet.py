"""mobilenetv2 in pytorch
[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

USE_DECOMPOSER = False


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


class CNNdecomposer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        groups=1,
        components=4,
    ):
        super().__init__()

        self.components = components

        alpha_shape = [1, out_channels] + [1] * 2  # hegith & width
        self.alpha_list = nn.ParameterList()
        self.cnn_list = nn.ModuleList()
        # self.bnn_list = nn.ModuleList()
        for _ in range(components):
            # self.bnn_list.append(nn.BatchNorm2d(out_channels))
            self.alpha_list.append(nn.Parameter(torch.ones(*alpha_shape)))
            self.cnn_list.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                )
            )

    def decompose(self, residual):
        residuals = []
        alpha_list = []

        for _ in range(self.components):
            alpha = torch.mean(residual ** 2, dim=[1, 2, 3]) ** (0.5)
            im = torch.tanh(residual)  # .sign()
            residuals.append(im)
            alpha = alpha[..., None, None, None]
            alpha_list.append(alpha)
            residual = residual - im * alpha.expand_as(residual)  #

        # print(alpha_list)
        # print(residual)
        return residuals, alpha_list

    def forward(self, x):
        residuals, alpha_list = self.decompose(x)

        out = 0
        x = 0
        for r, a, cnn, in zip(residuals, alpha_list, self.cnn_list,):
            o = cnn(r)
            out += o * a
        return out


def make_cnn(
    in_planes, planes, kernel_size, stride=1, padding=0, bias=True, groups=1
):
    if USE_DECOMPOSER:
        return CNNdecomposer(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
    else:
        return nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            make_cnn(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            make_cnn(
                in_channels * t,
                in_channels * t,
                3,
                stride=stride,
                padding=1,
                groups=in_channels * t,
            ),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            make_cnn(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2(nn.Module):
    def __init__(self, class_num=100):
        super().__init__()

        self.first_conv = nn.Conv2d(3, 32, 1, padding=1)
        self.first_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            make_cnn(320, 1280, 1), nn.BatchNorm2d(1280), nn.ReLU6(inplace=True)
        )

        self.conv_final = make_cnn(1280, class_num, 1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.relu(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv_final(x)
        x = x.view(x.size(0), -1)

        return {"preds": x}

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)


def mobilenetv2():
    return MobileNetV2()
