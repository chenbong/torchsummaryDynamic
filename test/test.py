import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryDynamic import summary

class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, us=[False, False]):
        super(USConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.width_mult = None
        self.us = us

    def forward(self, inputs):
        in_channels = inputs.shape[1] // self.groups if self.us[0] else self.in_channels // self.groups
        out_channels = int(self.out_channels * self.width_mult) if self.us[1] else self.out_channels

        weight = self.weight[:out_channels, :in_channels, :, :]
        bias = self.bias[:out_channels] if self.bias is not None else self.bias

        y = F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return y

model = nn.Sequential(
    USConv2d(3, 32, 3, us=[True, True]),
)

# width_mult=1.0
model.apply(lambda m: setattr(m, 'width_mult', 1.0))
summary(model, torch.zeros(1, 3, 224, 224))

# width_mult=0.5
model.apply(lambda m: setattr(m, 'width_mult', 0.5))
summary(model, torch.zeros(1, 3, 224, 224))

