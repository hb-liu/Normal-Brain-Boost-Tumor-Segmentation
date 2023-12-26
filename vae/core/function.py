import os
import torch
import numpy as np
import torchvision.utils as vutils
from utils.utils import AverageMeter
from utils.utils import set_requires_grad
from models.densenet import reparameterize
from core.loss import kl_loss, rec_loss, dis_loss, GANLoss

def train(enc, dec, dis, optim_enc, optim_dec, optim_dis, dataloader, logger, config, epoch):
    enc.train()
    dec.train()
    print_freq = 10

    lkls = AverageMeter()
    lrecs = AverageMeter()
    lgans = AverageMeter()
    ldiss = AverageMeter()

    beta_kl = 1.0
    beta_rec = 10.0
    beta_gan = 10.0
    gan_loss = GANLoss()

    vae_iter = 5
    num_iter = config.TRAIN.NUM_BATCHES
    scaler = torch.cuda.amp.GradScaler()
    for idx in range(num_iter):
        data_dict = next(dataloader)
        img = data_dict['data'].cuda()
        bs = img.shape[0]

        with torch.cuda.amp.autocast():
            for i in range(vae_iter):
                mu, logvar = enc(img)
                z = reparameterize(mu, logvar)
                rec = dec(z)
                zp = torch.randn_like(z)
                psu = dec(zp)

                set_requires_grad([dis], False)
                lrec = rec_loss(rec, img)
                lkl = kl_loss(mu, logvar)
                lgan = gan_loss(dis(psu), True) + gan_loss(dis(rec), True)
                loss = beta_rec * lrec + beta_kl * lkl + beta_gan * lgan

                lrecs.update(lrec.item(), bs)
                lkls.update(lkl.item(), bs)
                lgans.update(lgan.item(), bs)

                optim_enc.zero_grad()
                optim_dec.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim_enc)
                scaler.step(optim_dec)
                scaler.update()

            set_requires_grad([dis], True)
            ldis = dis_loss(gan_loss, dis, img, rec) + dis_loss(gan_loss, dis, img, psu)
            optim_dis.zero_grad()
            scaler.scale(ldis).backward()
            scaler.step(optim_dis)
            scaler.update()

            ldiss.update(ldis.item(), bs)

        if idx % print_freq == 0:
            info = "Epoch[{}]({}/{}): ".format(epoch, idx, num_iter)
            info += 'rec: {:.4f}({:.4f}), kl: {:.4f}({:.4f}), gan: {:.4f}({:.4f}), dis: {:.4f}({:.4f})'.format(lrecs.val, lrecs.avg, lkls.val, lkls.avg, lgans.val, lgans.avg, ldiss.val, ldiss.avg)
            logger.info(info)
            with torch.no_grad():
                img = torch.cat(torch.split(img, 1, 1))
                rec = torch.cat(torch.split(rec, 1, 1))
                psu = torch.cat(torch.split(psu, 1, 1))
                vutils.save_image(torch.cat([img, rec, psu], dim=0).data.cpu(), 'tmp/train.png', nrow=bs, scale_each=True, normalize=True)

@torch.no_grad()
def test(encoder, decoder, config):
    encoder.eval()
    decoder.eval()
    data = np.load(os.path.join(config.DATASET.ROOT, config.TEST.SAMPLE+'.npy'))
    # pad slice
    shape = np.array(data.shape[2:])
    pad_length = config.TRAIN.PATCH_SIZE - shape
    pad_left = pad_length // 2
    pad_right = pad_length - pad_length // 2
    pad_left = np.clip(pad_left, 0, pad_length)
    pad_right = np.clip(pad_right, 0, pad_length)
    data = np.pad(data, ((0, 0), (0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))
    image = torch.from_numpy(data[:-1]).cuda(device=0).permute(1, 0, 2, 3)
    bs = image.shape[0]
    # reconstruct
    mu, logvar = encoder(image)
    z = reparameterize(mu, logvar)
    rec = decoder(z)
    # save
    image = torch.cat(torch.split(image, 1, 1))
    rec = torch.cat(torch.split(rec, 1, 1))
    vutils.save_image(torch.cat([image, rec], dim=0).data.cpu(), 'tmp/test.jpg', nrow=bs, scale_each=True, normalize=True)