import torch
import os, pickle
import numpy as np
import torch.nn as nn
import statistics as stat
from core.config import config
from models.segnet import SEGNET
from medpy.metric.binary import *
from core.function import inference
import torch.backends.cudnn as cudnn
from utils.utils import create_logger, setup_seed

@torch.no_grad()
def inference(model, logger, config, dataset, metrics):
    model.eval()
    perfs = {}
    # calculate different kinds of metrics, e.g., dice, precision
    for metric in metrics:
        perfs[metric.__name__] = {'WT': [], 'ET': [], 'TC': []}
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
        # calculate metrics within brain region
        mask = image.permute(1, 0, 2, 3)[0].cpu().numpy() != 0
        # quantitative analysis
        for metric in metrics:
            predi = pred if metric.__name__ == 'hd95' else pred[mask]
            labeli = label if metric.__name__ == 'hd95' else label[mask]
            perfs[metric.__name__]['WT'].append(metric(predi > 0, labeli > 0))
            if 3 in label:
                perfs[metric.__name__]['ET'].append(metric(predi == 3, labeli == 3))
            if 2 in label:
                perfs[metric.__name__]['TC'].append(metric(predi >= 2, labeli >= 2))
    for metric in perfs.keys():
        et = perfs[metric]['ET']
        tc = perfs[metric]['TC']
        wt = perfs[metric]['WT']
        logger.info(f'------------ {metric} ------------')
        logger.info(f'ET mean / std: {stat.mean(et)} / {stat.stdev(et)}')
        logger.info(f'TC mean / std: {stat.mean(tc)} / {stat.stdev(tc)}')
        logger.info(f'WT mean / std: {stat.mean(wt)} / {stat.stdev(wt)}')

def main():
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = SEGNET(inc=4, outc=4, midc=16, stages=4)
    model = nn.DataParallel(model, config.TRAIN.DEVICES).cuda()
    model.load_state_dict(torch.load('results/vae/model_best.pth'))

    logger = create_logger('log', 'test.log')
    inference(model, logger, config, dataset='test', metrics=[dc, jc, hd95, sensitivity, precision, specificity])

if __name__ == '__main__':
    main()
