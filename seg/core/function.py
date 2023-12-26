import torch
import os, pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from medpy.metric.binary import dc
from utils.utils import AverageMeter
from torchvision.utils import save_image

def train(model, train_generator, optimizer, criterion, logger, config, epoch):
    segnet, sims = model
    segnet.train()
    optim_seg, optim_sim = optimizer
    print_freq = 10
    lsegs = AverageMeter()
    lsims = AverageMeter()
    scaler = torch.cuda.amp.GradScaler()
    num_iter = config.TRAIN.NUM_BATCHES
    for i in range(num_iter):
        data_dict = next(train_generator)
        image = data_dict['data'].cuda()
        labels = data_dict['label']
        labels = [label.cuda() for label in labels]

        with torch.cuda.amp.autocast():
            outs, embs = segnet(image)
            # simsiam
            lsim = 0
            for k in range(len(embs)):
                g1, x1 = embs[k]
                label = labels[k]
                fore = (image[:, 0:1] != 0).float()
                fore = F.interpolate(fore, size=label.shape[2:])
                mask = (label == 0) * fore  # foreground non-tumor region
                lsim += sims[k](g1, x1, mask).mean()
            lseg = criterion(outs, labels)
            loss = lseg + .1 * lsim
        lsegs.update(lseg.item(), config.TRAIN.BATCH_SIZE)
        lsims.update(lsim.item(), config.TRAIN.BATCH_SIZE)

        optim_seg.zero_grad()
        optim_sim.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optim_seg)
        torch.nn.utils.clip_grad_norm_(segnet.parameters(), 12)
        scaler.step(optim_seg)
        scaler.step(optim_sim)
        scaler.update()

        if i % print_freq == 0:
            msg = 'epoch: [{0}][{1}/{2}]\t' \
                'lseg {lseg.val:.3f} ({lseg.avg:.3f})\t' \
                'lsim {lsim.val:.3f} ({lsim.avg:.3f})'.format(
                    epoch, i, num_iter,
                    lseg = lsegs,
                    lsim = lsims
                )
            logger.info(msg)
            bs = image.shape[0]
            image = torch.cat(torch.split(image, 1, 1))
            label = torch.cat(torch.split(labels[-1], 1, 1))
            out = torch.argmax(torch.softmax(outs[-1], 1), dim=1, keepdim=True)
            out = torch.cat(torch.split(out, 1, 1))
            save_image(torch.cat([image, label, out], dim=0).data.cpu(), f'tmp/train.png', nrow=bs, scale_each=True, normalize=True)

@torch.no_grad()
def inference(model, logger, config, dataset):
    model.eval()
    perfs = {'WT': AverageMeter(), 'ET': AverageMeter(), 'TC': AverageMeter()}
    nonline = nn.Softmax(dim=1)
    with open(os.path.join(config.DATASET.ROOT, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
    valids = splits[dataset]
    for name in valids:
        data = np.load(os.path.join(config.DATASET.ROOT, name+'.npy'))
        # pad slice
        shape = np.array(data.shape[2:])
        pad_length = config.TRAIN.PATCH_SIZE - shape
        pad_left = pad_length // 2
        pad_right = pad_length - pad_length // 2
        pad_left = np.clip(pad_left, 0, pad_length)
        pad_right = np.clip(pad_right, 0, pad_length)
        data = np.pad(data, ((0, 0), (0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))
        # run inference
        image = torch.from_numpy(data[:-1]).permute(1, 0, 2, 3).cuda()
        label = data[-1]
        with torch.cuda.amp.autocast():
            out = model(image)
            out = nonline(out)
        pred = torch.argmax(out, dim=1).cpu().numpy()
        # quantitative analysis
        perfs['WT'].update(dc(pred > 0, label > 0))
        if 3 in label:
            perfs['ET'].update(dc(pred == 3, label == 3))
        if 2 in label:
            perfs['TC'].update(dc(pred >= 2, label >= 2))
    for c in perfs.keys():
        logger.info(f'class {c} dice mean: {perfs[c].avg}')
    logger.info('------------ ----------- ------------')
    perf = np.mean([perfs[c].avg for c in perfs.keys()])
    return perf