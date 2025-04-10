'''
Onet based on vanilla Unet (4 layer downsampling and 4 layer upsampling with normal CNN structures).
This model is unified for both the cloud segmentation and object segmentation.

based on the file 'Onet_L4H4_Dot_vanilla_20240503.py'
Created by ZhouYi@Linghai_Dalian on 20240606
'''


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import argparse
import utils_20231218 as uti
import pickle
from datetime import datetime
from skimage.transform import resize
from tqdm import tqdm

#import datasets.dsb_nucleus4onet_20230306 as dsb_model   # load dsb_datasets.
import configs.config_tip2022_20230411    as conf_model  # load configuration from '.yml' file.
#import evaluation_20230429                as eval_model  # load evaluation function.
import dataloader.simbg4onet_20230209       as simbg_model # load simbg_datasets.
import pandas as pd
import gc
import glob
import logging

torch.manual_seed(1981)
np.random.seed(1981)
torch.set_default_dtype(torch.float32)

'''
DoubleConv, Down,Up major blocks of Unet. 
'''
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            #nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, binit=False, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512, 256 // factor, bilinear)
        self.up3   = Up(256, 128 // factor, bilinear)
        self.up4   = Up(128, 64, bilinear)
        #self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        if binit:
            self._initialize_weights()

    def _initialize_weights(self, mode='fan_in'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode=mode, nonlinearity='relu') #relu
                # nn.init.kaiming_normal_(
                #     m.weight, mode=mode, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                # assert (m.track_running_stats == self.batchnorm_track)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # global feature

        y4 = self.up1(x5, x4)
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)
        return x1, y1  # local and global respresentation of features in DNN.

import copy
class Onet(nn.Module):
    def __init__(self, in_chns=1, binit=False, bshare=True):
        super(Onet, self).__init__()
        # self.topu = UNet(n_channels=inchannels, n_classes=1, bilinear=True) # top u model
        # self.dwnu = UNet(n_channels=inchannels, n_classes=1, bilinear=True) # down u model

        self.topu = UNet(n_channels=in_chns, n_classes=1, bilinear=False, binit=binit)  # top u model
        if bshare: # weight-share version.
            self.dwnu = self.topu
        else:
            self.dwnu = UNet(n_channels=in_chns, n_classes=1, bilinear=False, binit=binit)

        #self.dwnu = UNet(n_channels=1, n_classes=1, bilinear=False, binit=binit)  # down u model
        #self.dwnu = copy.deepcopy(self.topu)#UNet(n_channels=1, n_classes=1, bilinear=False, binit=binit)  # down u model
        # Define the softmax layer
        self.softmax = nn.Softmax2d()
        self.bias= 0 # background bias [0,1]. 0 means no bias, 1 means all background

    def forward(self, X):
        Lt, Ht = self.topu(X)
        Vt = torch.einsum("bpxy,bpxy->bxy", Lt, Ht)  # v shape as [B, H, W]
        Vt = Vt.unsqueeze(dim=1)  # v in [B, 1, H, W] # global feature at the decoder end

        # Ld, Hd = self.dwnu(1 - X + self.bias)
        Xd = torch.clip(1 - X + self.bias, 0, 1)
        Ld, Hd = self.dwnu(Xd)
        Vd = torch.einsum("bpxy,bpxy->bxy", Ld, Hd)  # v shape as [B, H, W]
        Vd = Vd.unsqueeze(dim=1)  # v in [B, 1, H, W] # global feature at the decoder end

        V = torch.concat([Vt, Vd], dim=1)

        #V = V/np.sqrt(64) # divide the scale of the logits

        S = self.softmax(V)  # gradient  backpropagation are needed, so not use onet.get_label()
        #Y = onet.predict_label(S)
        return Lt, Vt, Ld, Vd, S

    def predict_label(self, S):
        '''
        get predict labels from the classification output.
        :param S: class probability map.
        :return: Y : predict label map.
        '''
        with torch.no_grad():
            assert( S.dim() == 4 )
            Y = torch.argmax(S, dim=1)  # label [B, h, w]
        return Y

    def get_label(self, Vt, Vd):
        '''
        get predict labels from the classification output.
        :param Vt: global feature of top unet.
        :param Vd: global feature of down unet.
        :return:
        '''
        # Note that Vt and Vd may be in different order if they're not normalized.
        with torch.no_grad():
            # norm_vt = uti.tensor_normal_per_frame(Vt)
            # norm_vd = uti.tensor_normal_per_frame(Vd)
            # V = torch.concat([norm_vt, norm_vd], dim=1)
            V = torch.concat([Vt, Vd], dim=1)
            V = self.softmax(V)
            pred_labels = torch.argmax(V, dim=1)  # label [B, h, w]
            return pred_labels, V

    def jensen_shannon_divergence(self, Li, Si, Sprime):
        '''
        self inner product of L and S for the same samples, to compute the left part of JSD eq.
        Sprime is computed by the adversarial input.
        :param L: local feature of the encoder part in Unet
        :param S: class probability of the original input
        :param Sprime: class probability of the adversarial input
        :return: JSD value of two probability distributions. jsd(L,S)
        '''
        assert (Li.dim() == 4 and Si.dim() == 4 and Sprime.dim() == 4)
        LS = torch.einsum("bpxy,bpxy->bxy", Li, Si)       # for jsd's left part
        LSp= torch.einsum("bpxy,bpxy->bxy", Li, Sprime)  # for jsd's right part
        jsd = -1 * self.log1pexp(-1 * LS).mean() - self.log1pexp(LSp).mean()
        assert (torch.isnan(jsd) == False)
        return jsd

    def log1pexp(self, x):
        '''
        compute numerically stable log(1+exp(x))
        based on the paper:
        https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
        :param x:
        :return:
        '''
        x[x <= -37.] = torch.exp(x[x <= -37.])
        idx = (x > -37) * (x <= 18.)
        x[idx] = torch.log(1 + torch.exp(x[idx]))
        idx = (x > 18.) * (x < 33.3)
        x[idx] = x[idx] + torch.exp(-x[idx])
        # x[x>33.3] remain unchanged
        return x

    def compute_loss(self, Lt, St, Ld, Sd):
        #enscapulation for ablation
        # Note that getattr() normally throws exception when the attribute doesn't exist.
        # However, if you specify a default value (None, in this case), it will return that instead.
        # nce = getattr(self, "nce_loss", None)
        # if callable(nce):
        #     nce_loss = nce(Lt, St, Ld, Sd)
        #     return nce_loss

        jsd = getattr(self, "jensen_shannon_divergence", None) #comment jensen_shannon_divergence when use nce_loss
        if callable(jsd):
            jsd_top = jsd(Lt, St, Sd)
            jsd_dwn = jsd(Ld, Sd, St)
            jsd_loss= -(jsd_top + jsd_dwn)/2
            return jsd_loss