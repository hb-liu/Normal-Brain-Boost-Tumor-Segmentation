from easydict import EasyDict as edict

config = edict()
config.NUM_WORKERS = 8
config.MODEL_DIR = 'experiments'
config.DEVICES = [0, 1]
config.SEED = 12345

config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = False    # True leads to unexpected errors and slows training and inference

config.DATASET = edict()
config.DATASET.ROOT = '../data/processed'

config.TRAIN = edict()
config.TRAIN.LR = 2e-4
config.TRAIN.BATCH_SIZE = 32
config.TRAIN.PATCH_SIZE = (224, 224)
config.TRAIN.NUM_BATCHES = 250
config.TRAIN.EPOCH = 201

config.TEST = edict()
config.TEST.SAMPLE = 'BraTS2021_00351'