import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out

class CNNResNet(nn.Module):
    def __init__(self):
        super(CNNResNet, self).__init__()
        
        # 输入通道为3，输出为64，stride为2进行第一次下采样
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 下采样 x2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 继续下采样 x2
        # 总共4倍下采样

        # ResNet Block 1
        self.layer1 = self._make_layer(64, 64, stride=1)  # 保持大小不变
        
        # ResNet Block 2
        self.layer2 = self._make_layer(64, 128, stride=1)  # 不进行下采样
        
        # ResNet Block 3
        self.layer3 = self._make_layer(128, 128, stride=1)  # 保持大小不变
        
        # 全局平均池化，将特征映射到固定维度
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, in_channels, out_channels, stride=1):
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Conv Layer with downsampling
        x = self.conv1(x)  # 下采样 x2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 下采样 x2 -> 总共 x4

        # ResNet Layers
        x = self.layer1(x)  # 尺寸保持不变
        x = self.layer2(x)  # 尺寸保持不变
        x = self.layer3(x)  # 尺寸保持不变
        

        
        return x

if __name__ == '__main__':
    # 测试网络
    model = CNNResNet()
    x = torch.randn(1, 3, 896, 896)  # 输入大小为 (1, 3, 224, 224)
    output = model(x)
    print(output.shape)  # 输出的形状应为 (1, 128)
