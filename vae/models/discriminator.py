import torch.nn as nn

class ck(nn.Module):
    def __init__(self, inc, outc, stride, norm):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, 3, stride, 1)
        self.norm = nn.BatchNorm2d(outc) if norm else nn.Identity()
        self.relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class PatchDiscriminator(nn.Module):
    def __init__(self, inc=4, midc=8) -> None:
        super().__init__()
        self.conv1 = ck(inc, midc, 2, False)
        self.conv2 = ck(midc, midc*2, 2, True)
        self.conv3 = ck(midc*2, midc*4, 2, True)
        self.conv4 = nn.Conv2d(midc*4, 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x