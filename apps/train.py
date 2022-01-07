import time
# from matplotlib import pyplot as plt

from torchvision.utils import save_image
import torch

from app import Init, mask_op


def train_op(alpha, beta, checkpoints_dir,
             gpu_ids, fine_size,
             epochs, display_freq,
             save_epoch_freq, *,
             which_epoch=None):
    init = Init(fine_size, is_train=True)
    maskset = init.loader.maskset()  # mask 数据集

    opt = init.opt(alpha, beta, checkpoints_dir, gpu_ids,
                   which_epoch=which_epoch)

    model = init.model(opt)
    image_save_dir = init.create_image_save_dir(model)
    # 训练阶段
    total_steps = 0
    iter_start_time = time.time()
    for epoch in range(which_epoch+1, epochs):
        print(f"epoch: {epoch} \n")
        epoch_start_time = time.time()
        epoch_iter = 0
        # 初始化数据集
        trainset = init.loader.dataset()  # 训练集

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
                    f"Epoch/total_steps/alpha-beta/time: {epoch}/{total_steps}/{alpha}-{beta}/{t}", dict(errors))
        if epoch % save_epoch_freq == 0:
            print(
                f'保存模型 Epoch {epoch}, iters {total_steps} 在 {model.save_dir}')
            model.save(epoch)
        print(
            f'Epoch/Epochs {epoch}/{epochs-1} 花费时间：{time.time() - epoch_start_time}s')
        model.update_learning_rate()
