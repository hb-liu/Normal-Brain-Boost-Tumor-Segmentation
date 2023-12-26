import os
import pickle
import numpy as np
from collections import OrderedDict
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase


class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, data, batch_size, patch_size):
        super().__init__(data, batch_size, None)
        self.patch_size = patch_size
    
    def generate_train_batch(self):
        # random select data
        sels = np.random.choice(list(self._data.keys()), self.batch_size, True)
        # read data, form slice
        images, labels = [], []
        for i, name in enumerate(sels):
            data = np.load(self._data[name]['path'])
            # select slice w/o tumor
            locs = np.unique(np.where(data[-1] != 0)[0]).tolist()
            locs = [x for x in range(data.shape[1]) if x not in locs]
            sel_idx = np.random.choice(locs)
            data = data[:, sel_idx]
            # pad slice
            shape = np.array(data.shape[1:])
            pad_length = self.patch_size - shape
            pad_left = pad_length // 2
            pad_right = pad_length - pad_length // 2
            pad_left = np.clip(pad_left, 0, pad_length)
            pad_right = np.clip(pad_right, 0, pad_length)
            data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))
            images.append(data[:-1])
            labels.append(data[-1:])
        image = np.stack(images)
        label = np.stack(labels)
        return {'data': image, 'label': label}

def get_trainloader(config):
    with open(os.path.join(config.DATASET.ROOT, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
    names = splits['train']
    dataset = OrderedDict()
    for name in names:
        dataset[name] = OrderedDict()
        dataset[name]['path'] = os.path.join(config.DATASET.ROOT, name+'.npy')
        with open(os.path.join(config.DATASET.ROOT, name+'.pkl'), 'rb') as f:
            dataset[name]['locs'] = pickle.load(f)
    return DataLoader2D(dataset, config.TRAIN.BATCH_SIZE, config.TRAIN.PATCH_SIZE)