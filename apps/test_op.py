from torchvision.utils import save_image
import torch
from pathlib import Path
from matplotlib import pyplot as plt

from tools.file import mkdir
from app import Init, mask_op, array2image


class Test:
    def __init__(self, fine_size) -> None:
        self.init = Init(fine_size, is_train=False)
        self.maskset = self.init.loader.maskset()  # mask 数据集

    def valset(self):
        valset = self.init.loader.dataset()  # 验证集
        for image, path in valset:
            yield image, Path(path[0]).name

    def __iter__(self):
        valset = self.valset()
        maskset = self.maskset
        for (image, name), mask in zip(valset, maskset):
            yield image, name, mask

    def model(self, alpha, beta,
              checkpoints_dir, gpu_ids,
              *, which_epoch=-1):
        opt = self.init.opt(alpha, beta, checkpoints_dir, gpu_ids,
                            which_epoch=which_epoch)
        return self.init.model(opt)

    def run(self, epoch, image, image_name, mask, model, save_dir):
        save_dir = f'{save_dir}/{model.name}'
        mkdir(save_dir)
        mask = mask_op(mask)
        model.set_input(image, mask)
        model.set_gt_latent()
        model.test()
        real_A, real_B, fake_B = model.get_current_visuals()
        pic = (torch.cat([real_A, real_B, fake_B], dim=0) + 1) / 2.0

        save_image(pic, f"{save_dir}/{image_name}-epoch{epoch}.jpg", nrow=1)

        if epoch % 50 == 0:
            plt.figure(figsize=(6, 8))
            plt.subplot(1, 3, 1)
            x = array2image(real_A[0])
            plt.imshow(x)
            x = array2image(real_B[0])
            plt.subplot(1, 3, 2)
            plt.imshow(x)
            x = array2image(fake_B[0])
            plt.subplot(1, 3, 3)
            plt.imshow(x)
            plt.show()
