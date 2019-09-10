#    UNet_net.py : CNN architecture accompanying publication "Classification of prostate cancer on MRI: Deep learning vs. clinical PI-RADS assessment", Patrick Schelb, Simon Kohl, Jan Philipp Radtke MD, Manuel Wiesenfarth PhD, Philipp Kickingereder MD, Sebastian Bickelhaupt, Tristan Anselm Kuder PhD, Albrecht Stenzinger, Markus Hohenfellner MD, Heinz-Peter Schlemmer MD, PhD, Klaus H. Maier-Hein PhD, David Bonekamp MD, Radiology, [manuscript accepted for publication]
#    Copyright (C) 2019  German Cancer Research Center (DKFZ)

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

#    contact: David Bonekamp, MD, d.bonekamp@dkfz-heidelberg.de

__author__  = "German Cancer Research Center (DKFZ)"


import torch
import torch.nn as nn
import torch.nn.functional as F

# modified Unet implementation from https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208

BN_EPS = 1e-5

class UNetPytorch (nn.Module):
    def __init__(self, in_shape):
        super(UNetPytorch, self).__init__()
        C, H, W = in_shape

        self.down2 = StackEncoder(C, 64, kernel_size=3)
        self.down3 = StackEncoder(64, 128, kernel_size=3)
        self.down4 = StackEncoder(128, 256, kernel_size=3)
        self.down5 = StackEncoder(256, 512, kernel_size=3)
        self.down6 = StackEncoder(512, 1024, kernel_size=3)

        self.center = nn.Sequential(
            ConvRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1),
            ConvRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1),
            ConvRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1))

        self.up6 = StackDecoder(1024, 1024, 512, kernel_size=3)
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)
        self.up2 = StackDecoder(64, 64, 32, kernel_size=3)

        self.classify = nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, input):
        out = input

        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)
        down6, out = self.down6(out)
        pass

        out = self.center(out)

        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)

        return out


class ConvRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_relu=True, is_bn=True):
        super(ConvRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        if is_relu is False: self.relu = None
        if is_bn is False: self.bn = None

    def forward(self, input):
        convoluted = self.conv(input)
        if self.relu is not None: convoluted = self.relu(convoluted)
        if self.bn is not None: convoluted = self.bn(convoluted)
        return convoluted

class StackEncoder (nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(StackEncoder, self).__init__()
        padding=(kernel_size-1)//2

        self.encode = nn.Sequential(
            ConvRelu2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvRelu2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvRelu2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1))

    def forward(self, input):
        encoded = self.encode(input)
        max_pooled = F.max_pool2d(encoded, kernel_size=2, stride=2)

        return encoded, max_pooled


class StackDecoder (nn.Module):
    def __init__(self, in_channels_down, in_channels, out_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding=(kernel_size-1)//2

        self.decode = nn.Sequential(
            ConvRelu2d(in_channels_down + in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvRelu2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvRelu2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1))

    def forward(self, down_input, input):
        N, C, H, W = down_input.size()
        upsampled = F.interpolate(input, size=(H,W) ,mode='bilinear', align_corners=True)
        upsampled = torch.cat([upsampled, down_input], 1)
        decoded = self.decode(upsampled)
        return decoded

