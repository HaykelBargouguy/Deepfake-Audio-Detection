import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .attention import SelfAttention

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, downsample=None):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.downsample(out) if self.downsample is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class ResNetAttention(nn.Module):
    """
    ResNet backbone integrated with Custom Self-Attention.
    Optimized for extracting spectrogram/MFCC/LFCC spatial features.
    """
    def __init__(self, enc_dim=256, nclasses=2):
        super(ResNetAttention, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()

        self.layer1 = self._make_layer(PreActBlock, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(PreActBlock, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(PreActBlock, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(PreActBlock, 512, num_blocks=2, stride=2)

        self.conv5 = nn.Conv2d(512, 256, kernel_size=(5, 3), stride=(1, 1), padding=(0, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.attention = SelfAttention(256)
        
        self.fc = nn.Linear(256 * 2, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, nclasses)

        self.initialize_params()

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(self.bn1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        
        batch_size, channels, freq, time = x.shape
        x = self.activation(self.bn5(x)).view(batch_size, channels, -1)
        
        stats = self.attention(x.permute(0, 2, 1).contiguous())
        
        feat = self.fc(stats)
        mu = self.fc_mu(feat)
        return feat, mu