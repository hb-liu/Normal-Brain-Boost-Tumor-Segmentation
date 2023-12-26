import torch.nn as nn
import torch.nn.functional as F

class Proj(nn.Module):
    def __init__(self, inc, midc, outc) -> None:
        super().__init__()
        self.ln1 = nn.Sequential(
            nn.Linear(inc, midc),
            nn.BatchNorm1d(midc),
            nn.ReLU()
        )
        self.ln2 = nn.Sequential(
            nn.Linear(midc, midc),
            nn.BatchNorm1d(midc),
            nn.ReLU()
        )
        self.ln3 = nn.Sequential(
            nn.Linear(midc, outc),
            nn.BatchNorm1d(outc)
        )
    def forward(self, x):
        x = x.permute(1, 0, 2, 3).flatten(1).T
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        return x

class Pred(nn.Module):
    def __init__(self, inc, midc, outc) -> None:
        super().__init__()
        self.ln1 = nn.Sequential(
            nn.Linear(inc, midc),
            nn.BatchNorm1d(midc),
            nn.ReLU()
        )
        self.ln2 = nn.Linear(midc, outc)
    def forward(self, x):
        x = self.ln1(x)
        x = self.ln2(x)
        return x
    
class SimSiam(nn.Module):
    def __init__(self, inc, midc) -> None:
        super().__init__()
        self.proj = Proj(inc, midc, inc)
        self.pred = Pred(inc, midc, inc)
    def D(self, p, z):
        return - F.cosine_similarity(p, z.detach(), dim=-1)
    def forward(self, g, x, mask):
        mask = mask.permute(1, 0, 2, 3).flatten(1)[0]
        v1, v2 = self.proj(g), self.proj(x)
        p1, p2 = self.pred(v1), self.pred(v2)
        lsim = self.D(p1, v2) + self.D(p2, v1)
        lsim = lsim * mask
        return lsim.mean()