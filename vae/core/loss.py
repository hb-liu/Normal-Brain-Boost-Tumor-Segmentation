import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

def kl_loss(mu, logvar) -> Tensor:
    axes = list(range(1, len(mu.shape)))
    kl = -0.5 * (1 + logvar - logvar.exp() - mu.pow(2)).sum(axes)
    return kl.mean()

def rec_loss(rec, target):
    target, rec = target.flatten(1), rec.flatten(1)
    err = F.mse_loss(rec, target, reduction='none').sum(1)
    return err.mean()

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input).to(input.device)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def dis_loss(criterion, dis, real, fake):
    # real
    pred_real = dis(real)
    loss_D_real = criterion(pred_real, True)
    # fake
    pred_fake = dis(fake.detach())
    loss_D_fake = criterion(pred_fake, False)
    # combine
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    return loss_D