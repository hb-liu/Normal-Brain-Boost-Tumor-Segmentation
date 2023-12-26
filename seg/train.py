import torch
import torch.nn as nn
import torch.optim as optim
from core.config import config
from models.segnet import SEGNET
import torch.backends.cudnn as cudnn
from models.simsiam import SimSiam
from core.scheduler import PolyScheduler
from core.function import train, inference
from utils.utils import demodule_state_dict
from core.loss import DiceCELoss, MultiOutLoss
from dataset.dataloader import get_trainloader
from dataset.augmenter import get_train_generator
from utils.utils import save_checkpoint, create_logger, setup_seed

def main():
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    stages = config.MODEL.STAGES
    midc = config.MODEL.MIDCHANNEL
    # build segmentation network and simsiam network
    segnet = SEGNET(inc=4, outc=4, midc=midc, stages=stages)
    segnet = nn.DataParallel(segnet, config.TRAIN.DEVICES).cuda()
    sims = []
    chs = [midc//2 * 2**i for i in range(stages)][::-1]    # embedding dim of the fam module
    for ch in chs:
        sim = SimSiam(inc=ch, midc=ch//2)
        sims.append(nn.DataParallel(sim, config.TRAIN.DEVICES).cuda())
    # load pretrained vae reconstruction network
    vae_state_dict = torch.load(config.MODEL.VAE_STATE_DICT)
    segnet.module.normnet.vae['enc'].load_state_dict(demodule_state_dict(vae_state_dict['enc']))
    segnet.module.normnet.vae['dec'].load_state_dict(demodule_state_dict(vae_state_dict['dec']))
    # training stuff
    optim_seg = optim.SGD(segnet.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY, momentum=0.95, nesterov=True)
    sim_params = []
    for i in range(len(sims)):
        sim_params += (sims[i].parameters())
    optim_sim = optim.Adam(sim_params, lr=1e-4)
    sched_seg = PolyScheduler(optim_seg, t_total=config.TRAIN.EPOCH)
    sched_sim = PolyScheduler(optim_sim, t_total=config.TRAIN.EPOCH)
    # deep supervision
    scales = [1/2**i for i in range(stages)][::-1]
    criterion = MultiOutLoss(DiceCELoss(), weights=scales)
    trainloader = get_trainloader(config)
    train_generator = get_train_generator(trainloader, scales, num_workers=config.NUM_WORKERS)

    best_model = False
    best_perf = 0.0
    logger = create_logger('log', 'train.log')
    for epoch in range(config.TRAIN.EPOCH):
        logger.info('learning rate : {}'.format(optim_seg.param_groups[0]['lr']))
        
        train((segnet, sims), train_generator, (optim_seg, optim_sim), criterion, logger, config, epoch)
        sched_seg.step()
        sched_sim.step()
        perf = inference(segnet, logger, config, dataset='val')
        
        if perf > best_perf:
            best_perf = perf
            best_model = True
        else:
            best_model = False
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': segnet.state_dict(),
            'perf': perf
        }, best_model, config.OUTPUT_DIR, filename='checkpoint.pth')

if __name__ == '__main__':
    main()