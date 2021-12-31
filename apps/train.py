import time
from matplotlib import pyplot as plt

from torchvision.utils import save_image
import torch

from app import init, mask_op, array2image
from tools.file import mkdir
from utils.torch_loader_all import Loader
from tools.toml import load_option


# checkpoints_dir = 'D:/BaiduNetdiskWorkspace/result'
# gpu_ids = [0, 1]


def train_op(alpha, beta, fine_size,
             epochs,
             display_freq, save_epoch_freq,
             checkpoints_dir, gpu_ids=[0]):
    opt = load_option(f'../options-new/loader.toml')
    opt['fine_size'] = fine_size
    loader = Loader(**opt)
    model_name = f'CSAx-{fine_size}-{beta}-{alpha}'
    base_opt = load_option(f'../options-new/base.toml')
    base_opt['fine_size'] = fine_size
    model_opt = load_option('../options-new/train.toml')
    model_opt.update(
        {
            'checkpoints_dir': checkpoints_dir,
            'gpu_ids': gpu_ids
        }
    )
    model = init(model_name, beta, model_opt, base_opt)
    image_save_dir = model.save_dir / 'images'
    mkdir(image_save_dir)

    # 训练阶段
    start_epoch = 0
    total_steps = 0
    iter_start_time = time.time()
    iter_start_time = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        epoch_iter = 0
        epoch_iter = 0
        # 初始化数据集
        trainset = loader.trainset()  # 训练集
        maskset = loader.maskset()  # mask 数据集
        for (image, _), mask in zip(trainset, maskset):
            mask = mask_op(mask)
            total_steps += model.batch_size
            epoch_iter += model.batch_size
            # it not only sets the input data with mask,
            #  but also sets the latent mask.
            model.set_input(image, mask)
            model.set_gt_latent()
            model.optimize_parameters()
            if total_steps % display_freq == 0:
                real_A, real_B, fake_B = model.get_current_visuals()
                # real_A=input, real_B=ground truth fake_b=output
                pic = (torch.cat([real_A, real_B, fake_B], dim=0) + 1) / 2.0
                image_name = f"epoch{epoch}-{total_steps}-{alpha}.png"
                save_image(pic, image_save_dir/image_name, ncol=1)
            if total_steps % 100 == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / model.batch_size
                print(
                    f"Epoch/total_steps/alpha-beta: {epoch}/{total_steps}/{alpha}-{beta}", dict(errors))
        if epoch % save_epoch_freq == 0:
            print(
                f'保存模型 Epoch {epoch}, iters {total_steps} 在 {model.save_dir}')
            model.save(epoch)
        print(
            f'Epoch/Epochs {epoch}/{epochs-1} 花费时间：{time.time() - epoch_start_time}s')
        model.update_learning_rate()
