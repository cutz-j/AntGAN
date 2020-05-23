"""
adapted from https://github.com/1adrianb/face-alignment
adapted from https://github.com/protossw512/AdaptiveWingLoss
"""

from copy import deepcopy
from functools import partial

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.vgg import VGG
import imp

class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

class FAN(nn.Module):
    def __init__(self, args):
        super(FAN, self).__init__()

        self.vgg = VGG()
        if args.fname_vgg is not None:
            self.load_pretrained_weights(args.fname_ir, args.fname_vgg, args.device)

    def load_pretrained_weights(self, fname_ir, fname_w, device):
        MainModel = imp.load_source('MainModel', fname_ir)
        if torch.cuda.is_available():
            checkpoint = torch.load(fname_w)
        else:
            checkpoint = torch.load(fname_w, map_location=torch.device('cpu'))
    
        self.vgg.load_state_dict(checkpoint.state_dict(), strict=False)
        self.vgg.eval()
        self.vgg.to(device)

    def forward(self, x):
        outputs = self.vgg(x) # (bs, 511, )
        return outputs

    @torch.no_grad()
    def get_heatmap(self, x):
        ''' outputs 0-1 normalized heatmap '''
        x = F.interpolate(x, size=256, mode='bilinear')
        x_01 = x*0.5 + 0.5
        outputs = self(x_01) # (512, w, h)
        heatmaps = outputs[2][:, :3, :, :] # (bs, 3, w, h)
        return heatmaps