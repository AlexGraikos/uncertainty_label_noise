import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# Base convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # Block parameters
        self.kernel_size = 3
        self.stride = 1
        self.padding = (self.kernel_size - self.stride) // 2
        self.groups = 2
        self.slope = 0.01
        # Block layers
        self.conv1 = nn.Conv2d(in_channels, out_channels[0],
                               kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding)
        self.group_norm1 = nn.GroupNorm(self.groups, out_channels[0])
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1],
                               kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding)
        self.group_norm2 = nn.GroupNorm(self.groups, out_channels[1])
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2],
                               kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding)
        self.group_norm3 = nn.GroupNorm(self.groups, out_channels[2])

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = F.leaky_relu(self.group_norm1(x_out), self.slope)
        x_out = self.conv2(x_out)
        x_out = F.leaky_relu(self.group_norm2(x_out), self.slope)
        x_out = self.conv3(x_out)
        x_out = F.leaky_relu(self.group_norm3(x_out), self.slope)
        return x_out

# Baseline classification model
class BaselineCNN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(BaselineCNN, self).__init__()
        # Model parameters
        self.pool_kernel_size = 2
        self.pool_stride = 2
        self.p_dropoout = 0.25
        # Model layers
        self.conv_block1 = ConvBlock(in_channels, [128, 128, 128])
        self.max_pool1 = nn.MaxPool2d(kernel_size=self.pool_kernel_size, 
                                      stride=self.pool_stride)
        self.dropout1 = nn.Dropout(p=self.p_dropoout)

        self.conv_block2 = ConvBlock(128, [256, 256, 256])
        self.max_pool2 = nn.MaxPool2d(kernel_size=self.pool_kernel_size, 
                                      stride=self.pool_stride)
        self.dropout2 = nn.Dropout(p=self.p_dropoout)

        self.conv_block3 = ConvBlock(256, [512, 256, 128])
        self.max_pool3 = nn.MaxPool2d(kernel_size=self.pool_kernel_size, 
                                      stride=self.pool_stride)
        self.dropout3 = nn.Dropout(p=self.p_dropoout)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=2 * self.pool_kernel_size,
                                     stride=2 * self.pool_stride)
        self.dense = nn.Linear(128, n_classes)

    def forward(self, x):
        x_out = self.conv_block1(x)
        x_out = self.max_pool1(x_out)
        x_out = self.dropout1(x_out)
        x_out = self.conv_block2(x_out)
        x_out = self.max_pool2(x_out)
        x_out = self.dropout2(x_out)
        x_out = self.conv_block3(x_out)
        x_out = self.max_pool3(x_out)
        x_out = self.dropout3(x_out)
        x_out = self.avg_pool(x_out)
        x_out = torch.flatten(x_out, start_dim=1)
        x_out = self.dense(x_out)
        return x_out
  

if __name__ == '__main__':
    from data import *
    (train_loader, val_loader,
     train_clean_loader, val_clean_loader,
     test_loader) = load_cifar10()

    print('[*] Creating network...')
    baseline_net = BaselineCNN(in_channels=3, n_classes=10)
    images, labels = iter(train_loader).next()
    output = baseline_net(images)
    print('[*] Network output shape:', output.shape)
