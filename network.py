import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, c):
        super(SimpleNN, self).__init__()
        self.conv = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = x.permute(0, 2, 3, 1)
        return x


class SimpleResBlock(nn.Module):
    """
    A simple residual block for ResNet.
    The block consists of two convolutional layers with batch normalization and ReLU activation.
    The block also includes a skip connection to handle the case when the input and output dimensions do not match.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            stride (int): The stride for the convolutional layers.
        """
        super(SimpleResBlock, self).__init__()
        self.mlist = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if stride!=1 or in_channels!=out_channels else None
        self.relu = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        ori = x if self.res_conv==None else self.res_conv(x)
        out = self.relu(ori + self.mlist(x))
        return out


class DownSamplerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplerBlock, self).__init__()
        self.cnnblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.downsamplerblock = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        cnn_res = self.cnnblock(x)
        return cnn_res, self.downsamplerblock(cnn_res)

class UpSamplerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSamplerBlock, self).__init__()
        self.cnnblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(),
            nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()
        )
        self.upsamplerblock=nn.Sequential(
            nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.upsamplerblock(self.cnnblock(x))
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        channel = [in_channels, in_channels*2, in_channels*4]
        self.downsampler1 = DownSamplerBlock(channel[0], channel[1])
        self.downsampler2 = DownSamplerBlock(channel[1], channel[2])
        self.upsampler2 = UpSamplerBlock(channel[2], channel[2])
        self.upsampler1 = UpSamplerBlock(channel[2]*2, channel[1])
        self.outnet = nn.Sequential(
            nn.Conv2d(channel[1]*2, channel[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(),
            nn.Conv2d(channel[1], channel[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(),
            nn.Conv2d(channel[0], out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        ds1, x = self.downsampler1(x)
        ds2, x = self.downsampler2(x)
        x = self.upsampler2(x)
        x = torch.cat((x, ds2), dim=1)
        x = self.upsampler1(x)
        x = torch.cat((x, ds1), dim=1)
        x = self.outnet(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(ResNet, self).__init__()
        self.channels = [out_channels//2, out_channels//2, out_channels, out_channels*2, out_channels*2]
        self.prework = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(self.channels[0]),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resblocks1 = nn.Sequential(
            SimpleResBlock(self.channels[0], self.channels[1]),
            SimpleResBlock(self.channels[1], self.channels[2])
        )
        self.resblocks2 = nn.Sequential(
            self._make_layer(self.channels[2], self.channels[3], stride=2),
            self._make_layer(self.channels[3], self.channels[4], stride=2),
        )
        self.predict_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.channels[-1], num_classes)
        )

    def _make_layer(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            SimpleResBlock(in_channels, out_channels, stride=stride),
            SimpleResBlock(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.prework(x)
        x = self.resblocks1(x)
        # pred_x = None
        pred_x = self.resblocks2(x)
        pred_x = self.predict_layer(pred_x)
        return x, pred_x

class DeformableRegistrationNet(nn.Module):
    def __init__(self, img_channels, middle_channels):
        assert middle_channels>=4
        super(DeformableRegistrationNet, self).__init__()
        self.resnet = ResNet(in_channels=img_channels, out_channels=middle_channels, num_classes=10)
        self.unet = UNet(in_channels=middle_channels, out_channels=2)

    def forward(self, x):
        x, logits = self.resnet(x)
        x = self.unet(x)
        x = torch.tanh(x)
        x = x.permute(0, 2, 3, 1)
        return x, logits
        return x