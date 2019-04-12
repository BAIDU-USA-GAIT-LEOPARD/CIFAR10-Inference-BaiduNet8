'''BaiduNet-8 based on ResidualBlock
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ConvBN(nn.Module):
  def __init__(self, c_in, c_out):
    super(ConvBN, self).__init__()
    self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn = nn.BatchNorm2d(c_out)

  def forward(self, x):
    return F.relu(self.bn(self.conv(x)))

class Residual(nn.Module):
  def __init__(self, c_in, c_out):
    super(Residual, self).__init__()
    self.pre = ConvBN(c_in, c_out)
    self.conv_bn1 = ConvBN(c_out, c_out)
    self.conv_bn2 = ConvBN(c_out, c_out)

  def forward(self, x):
    x = self.pre(x)
    x = F.max_pool2d(x, 2)
    return self.conv_bn2(self.conv_bn1(x)) + x
   

class BaiduNet(nn.Module):
    def __init__(self, num_classes=10):
        super(BaiduNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.layer1 = Residual(64, 128)
        self.layer2 = Residual(128, 256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def BaiduNet8():

    return BaiduNet()


















