import time
# from pathlib import Path

# from random import randint
# from matplotlib import pyplot as plt

import torch as np

from models.CSA import CSA

from tools.file import mkdir



def array2image(x):
    x *= 255
    x = x.detach().cpu().numpy()
    return x.astype('uint8').transpose((1, 2, 0))

def mask_op(mask):
    mask = mask.cuda()
    mask = mask[0][0]
    mask = np.unsqueeze(mask, 0)
    mask = np.unsqueeze(mask, 1)
    mask = mask.byte()
    return mask


def init(model_name, beta, model_opt, base_opt):
    model_opt.update(base_opt)
    model_opt.update({'name': model_name}) # 设定模型名称
    model = CSA(beta, **model_opt)

    image_save_dir = model.save_dir / 'images'
    mkdir(image_save_dir)
    return model