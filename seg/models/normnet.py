import torch
import os, sys
import torch.nn as nn
sys.path.append(os.path.realpath('..'))
from vae.models.densenet import Encoder, Decoder, reparameterize

class Conv3x3(nn.Module):
    def __init__(self, inc, outc, stride):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, 3, stride, 1)
        self.bn = nn.BatchNorm2d(outc)
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class NORMNET(nn.Module):
    def __init__(self, inc, midc, stages):
        super().__init__()
        self.inc, self.stages = inc, stages
        # reconstruction network
        self.vae = nn.ModuleDict({'enc': Encoder(), 'dec': Decoder()})
        # input convolution
        self.inconv = Conv3x3(inc, midc, 1)
        # encoder
        self.enc = nn.ModuleList()
        stagec = midc
        for k in range(stages):
            self.enc.append(Conv3x3(stagec, stagec*2, 2))
            stagec *= 2
        # decoder
        self.dec = nn.ModuleList()
        for k in range(stages):
            self.dec.append(nn.ModuleList([
                nn.ConvTranspose2d(stagec, stagec//2, 2, 2),
                Conv3x3(stagec, stagec//2, 1)
            ]))
            stagec //= 2

    @torch.no_grad()
    def reconstruct(self, x):
        mu, logvar = self.vae['enc'](x)
        z = reparameterize(mu, logvar)
        rec = self.vae['dec'](z)
        return rec

    def forward(self, x):
        rec = self.reconstruct(x)
        x = self.inconv(rec)
        enc_feas = []
        for layer in self.enc:
            x = layer(x)
            enc_feas.append(x)
        fea = enc_feas[-1]
        dec_feas = []
        for i, (up, merge) in enumerate(self.dec):
            fea = up(fea)
            if i < len(enc_feas)-1:
                fea = merge(torch.cat([fea, enc_feas[-2-i]], 1))
            dec_feas.append(fea)
        return dec_feas