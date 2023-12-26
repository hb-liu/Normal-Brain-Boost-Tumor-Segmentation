import torch
import torch.nn as nn
from torchvision.models import densenet121
from torchvision.models.densenet import _DenseBlock

class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.Upsample(scale_factor=2.0))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        num_init_features = 64
        self.encoder = densenet121(pretrained=True)
        self.encoder.features[0] = nn.Conv2d(4, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Conv2d(1024, 2 * 1024, 3, 1, 1)
    
    def forward(self, x):
        # in conv
        x = self.encoder.features[0](x)
        x = self.encoder.features[1](x)
        x = self.encoder.features[2](x)
        x = self.encoder.features[3](x)
        # dense
        x = self.encoder.features[4](x)
        # transition
        x = self.encoder.features[5](x)
        # dense
        x = self.encoder.features[6](x)
        # transition
        x = self.encoder.features[7](x)
        # dense
        x = self.encoder.features[8](x)
        # transition
        x = self.encoder.features[9](x)
        # dense
        x = self.encoder.features[10](x)
        # norm
        x = self.encoder.features[11](x)
        
        x = self.fc(x)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar

def reparameterize(mu, logvar):
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, cdim=4, zdim=1024):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(zdim, 1024, 3, 1, 1),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            _Transition(1024, 512),
            _DenseBlock(num_layers=16, num_input_features=512, bn_size=4, growth_rate=32, drop_rate=0),

            _Transition(1024, 256),
            _DenseBlock(num_layers=8, num_input_features=256, bn_size=4, growth_rate=32, drop_rate=0),

            _Transition(512, 128),
            _DenseBlock(num_layers=4, num_input_features=128, bn_size=4, growth_rate=32, drop_rate=0),
        
            _Transition(256, 64),
            _DenseBlock(num_layers=2, num_input_features=64, bn_size=4, growth_rate=32, drop_rate=0),

            _Transition(128, 32),

            nn.Conv2d(32, cdim, 3, 1, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.decoder(x)
        return x
