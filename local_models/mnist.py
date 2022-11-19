import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNdecomposer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=False,
        components=3,
    ):
        super().__init__()

        self.components = components

        alpha_shape = [1, 1] + [1] * 2  # hegith & width
        self.alpha_list = nn.ParameterList()
        self.cnn_list = nn.ModuleList()
        # self.bnn_list = nn.ModuleList()
        for _ in range(components):
            # self.bnn_list.append(nn.BatchNorm2d(out_channels))
            self.alpha_list.append(
                nn.Parameter(torch.ones(*alpha_shape)) / out_channels
            )
            self.cnn_list.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=True,
                )
            )

    def decompose(self, residual):
        residuals = []
        alpha_list = []

        for _ in range(self.components):
            init = residual
            alpha = torch.mean(residual.abs(), dim=[1, 2, 3])
            sign = residual.sign()
            alpha = alpha[..., None, None, None]
            alpha_list.append(alpha)
            residual = residual - sign * alpha.expand_as(residual)  #

            im = (init - residual.detach()) / alpha.expand_as(residual)
            residuals.append(im)

        return residuals, alpha_list

    def forward(self, x):

        residuals, alpha_list = self.decompose(x)

        out = 0
        for r, a, cnn, in zip(residuals, self.alpha_list, self.cnn_list,):
            # print(a)
            o = cnn(r)
            out += o
        return out


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.bn = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 8, 5, stride=5, bias=True)

        alpha_shape = [1, 8] + [1] * (self.conv1.weight.dim() - 2)
        self.alpha_one = nn.Parameter(torch.ones(*alpha_shape))

        self.conv2 = nn.Conv2d(8, 8, 3, stride=1, bias=True)

        alpha_shape = [1, 8] + [1] * (self.conv1.weight.dim() - 2)
        self.alpha_two = nn.Parameter(torch.ones(*alpha_shape))

        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)  # * self.alpha_one
        x = F.relu(x)
        x = self.conv2(x)  # * self.alpha_two
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return {"preds": output}


class SimpleCNNDecomposed(nn.Module):
    def __init__(self):
        super(SimpleCNNDecomposed, self).__init__()
        self.bn = nn.BatchNorm2d(1)
        self.conv1 = CNNdecomposer(
            in_channels=1, out_channels=8, kernel_size=5, stride=5, bias=True
        )
        self.conv2 = CNNdecomposer(
            in_channels=8, out_channels=8, kernel_size=3, stride=1, bias=True
        )
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return {"preds": output}


def two_layer_cnn():
    return SimpleCNN()


def decomposed_cnn():
    return SimpleCNNDecomposed()
