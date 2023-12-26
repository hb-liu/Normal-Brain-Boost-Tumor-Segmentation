import torch
import torch.nn as nn
import torch.optim as optim
from core.config import config
from utils.utils import setup_seed
import torch.backends.cudnn as cudnn
from core.function import train, test
from models.densenet import Encoder, Decoder
from models.discriminator import PatchDiscriminator
from dataset.dataloader import get_trainloader
from utils.utils import create_logger, save_checkpoint
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

def main():
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    enc = Encoder()
    enc = nn.DataParallel(enc, device_ids=config.DEVICES).cuda()
    dec = Decoder()
    dec = nn.DataParallel(dec, device_ids=config.DEVICES).cuda()
    dis = PatchDiscriminator()
    dis = nn.DataParallel(dis, device_ids=config.DEVICES).cuda()
    optim_enc = optim.Adam(enc.parameters(), config.TRAIN.LR)
    optim_dec = optim.Adam(dec.parameters(), config.TRAIN.LR)
    optim_dis = optim.Adam(dis.parameters(), config.TRAIN.LR/10)
    sched_enc = optim.lr_scheduler.ExponentialLR(optim_enc, gamma=0.985)
    sched_dec = optim.lr_scheduler.ExponentialLR(optim_dec, gamma=0.985)
    sched_dis = optim.lr_scheduler.ExponentialLR(optim_dis, gamma=0.985)

    batch_generator = MultiThreadedAugmenter(
        data_loader=get_trainloader(config),
        transform=NumpyToTensor(keys=['data', 'label'], cast_to='float'),
        num_processes=config.NUM_WORKERS,
        pin_memory=True
    )

    output_dir = config.MODEL_DIR
    logger = create_logger('log', 'train.log')
    for epoch in range(config.TRAIN.EPOCH):
        train(enc, dec, dis, optim_enc, optim_dec, optim_dis, batch_generator, logger, config, epoch)
        sched_enc.step()
        sched_dec.step()
        sched_dis.step()
        logger.info('=> saving checkpoint to {}'.format(output_dir))
        test(enc, dec, config)

        save_checkpoint({
            'enc': enc.state_dict(),
            'dec': dec.state_dict()
        }, False, output_dir, filename='checkpoint.pth')

if __name__ == '__main__':
    main()