# import time
from pathlib import Path

# from random import randint
# from matplotlib import pyplot as plt

import torch as np

from models.CSA import CSA

from tools.file import mkdir
from tools.toml import load_option
from utils.torch_loader_all import Loader


def array2image(x, imtype='uint8'):
    x = x.detach().cpu().numpy()
    x = x.transpose((1, 2, 0))
    x /= 2.0 
    x *= 255.0
    return x.astype(imtype)


def mask_op(mask):
    mask = mask[0][0]
    mask = np.unsqueeze(mask, 0)
    mask = np.unsqueeze(mask, 1)
    mask = mask.byte()
    return mask


class Init:
    def __init__(self, fine_size, is_train=True) -> None:
        self.root = Path(__file__).parent.parent.as_posix()
        self.is_train = is_train
        self.fine_size = fine_size

    @property
    def loader(self):
        if not self.is_train:
            loader_path = f'{self.root}/options-new/loader-test.toml'
        else:
            loader_path = f'{self.root}/options-new/loader.toml'
        opt = load_option(loader_path)
        _loader = Loader(**opt, fine_size=self.fine_size,
                         is_train=self.is_train)
        return _loader

    def opt(self, alpha, beta, checkpoints_dir, gpu_ids, *,
            which_epoch=-1):
        model_name = f'CSAx-{self.fine_size}-{alpha}-{beta}'
        base_opt = load_option(f'{self.root}/options-new/base.toml')
        model_opt = load_option(f'{self.root}/options-new/model.toml')
        model_opt.update(base_opt)
        model_opt.update(
            {
                'beta': beta,
                'checkpoints_dir': checkpoints_dir,
                'gpu_ids': gpu_ids,
                'name': model_name,
                'fine_size': self.fine_size,
                'is_train': self.is_train,
                'which_epoch': which_epoch
            }
        )
        return model_opt

    def create_image_save_dir(self, model):
        image_save_dir = model.save_dir / 'images'
        mkdir(image_save_dir)
        return image_save_dir

    def model(self, opt):
        return CSA(**opt)
