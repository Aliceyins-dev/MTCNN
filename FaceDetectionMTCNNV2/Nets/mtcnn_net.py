import torch
import torch.nn as nn


class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseConvBlock, self).__init__()
        self.net_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1, 1, padding=0),
            nn.BatchNorm2d(in_channels//2),
            nn.PReLU(),
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size, stride, groups=in_channels//2),
            nn.BatchNorm2d(in_channels//2),
            nn.PReLU(),
            nn.Conv2d(in_channels//2, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.net_layer(x)


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.pnet_layer = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1, padding=0),  # 10
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 5
            DepthwiseConvBlock(10, 16, 3, 1),  # 3
            DepthwiseConvBlock(16, 32, 3, 1)   # 1
        )

        self.conv4_1 = nn.Conv2d(32, 1, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        x = self.pnet_layer(x)
        classify = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return classify, offset


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.rnet_layer = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1, padding=0),  # 22
            nn.PReLU(),
            nn.MaxPool2d(3, 2, padding=1),   # 11
            DepthwiseConvBlock(28, 48, 3, 1),  # 9
            nn.MaxPool2d(3, 2, padding=0),   # 4
            DepthwiseConvBlock(48, 64, 2, 1)   # 3
        )

        self.fcn_layer = nn.Linear(3*3*64, 128, bias=True)
        self.prelu = nn.PReLU()
        self.conv5_1 = nn.Linear(128, 1)
        self.conv5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.rnet_layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcn_layer(x)
        x = self.prelu(x)
        classify = torch.sigmoid(self.conv5_1(x))
        offset = self.conv5_2(x)
        return classify, offset


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.onet_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=0),  # 46
            nn.PReLU(),
            nn.MaxPool2d(3, 2, padding=1),  # 23

            DepthwiseConvBlock(32, 64, 3, 1),  # 21
            nn.MaxPool2d(3, 2, padding=0),  # 10

            DepthwiseConvBlock(64, 64, 3, 1),  # 8
            nn.MaxPool2d(2, 2, padding=0),  # 4
            DepthwiseConvBlock(64, 128, 2, 1)    # 3
        )

        self.fcn_layer = nn.Linear(3 * 3 * 128, 256, bias=True)
        self.prelu = nn.PReLU()
        self.conv6_1 = nn.Linear(256, 1)
        self.conv6_2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.onet_layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcn_layer(x)
        x = self.prelu(x)
        classify = torch.sigmoid(self.conv6_1(x))
        offset = self.conv6_2(x)
        return classify, offset
