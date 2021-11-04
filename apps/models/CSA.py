import torch
from collections import OrderedDict
from torch.autograd import Variable
from torch import nn

from torch.nn import functional as F
from .base_model import BaseModel
from . import networks
from .vgg16 import Vgg16

class ReliefCNN(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.h_conv = nn.Conv2d(in_units, units, (1, 3), padding=(0, 1))
        self.v_conv = nn.Conv2d(units, units,
                                kernel_size=(3, 1), padding=(1, 0))
        self.s_conv = nn.Conv2d(units, units,
                                kernel_size=(3, 3), padding=(1, 1))

    def forward(self, X):
        x = self.h_conv(X)
        y = self.v_conv(X)
        z = torch.exp(x) + torch.exp(y) + self.v_conv(x) + self.h_conv(y)
        x = self.s_conv(x)
        return F.relu(x)


class CSA(BaseModel):
    def __init__(self, beta, name, is_train, checkpoints_dir, gpu_ids,
                 lambda_A, gan_weight, cosis,
                 batch_size, mask_type, ngf, ndf,
                 which_model_netG, which_model_netP, which_model_netD, which_model_netF,
                 gan_type, continue_train, which_epoch, lr, beta1,
                 lr_policy, niter, niter_decay, lr_decay_iters, epoch_count, overlap,
                 fine_size, init_gain, input_nc, input_nc_g, output_nc, norm, use_dropout, init_type):
        super().__init__(name, is_train, checkpoints_dir, gpu_ids)
        self.beta = beta # test 时，设置为 1
        self.lambda_A = lambda_A
        self.gan_weight = gan_weight
        self.cosis = cosis
        self._initialize(batch_size, mask_type, ngf, ndf,
                         which_model_netG, which_model_netP, which_model_netD, which_model_netF,
                         gan_type, continue_train, which_epoch, lr, beta1,
                         lr_policy, niter, niter_decay, lr_decay_iters, epoch_count, overlap,
                         fine_size, init_gain, input_nc, input_nc_g, output_nc, norm, use_dropout, init_type)

    def _initialize(self, batch_size, mask_type, ngf, ndf,
                    which_model_netG, which_model_netP, which_model_netD, which_model_netF,
                    gan_type, continue_train, which_epoch, lr, beta1,
                    lr_policy, niter, niter_decay, lr_decay_iters, epoch_count, overlap,
                    fine_size, init_gain, input_nc, input_nc_g, output_nc, norm, use_dropout, init_type):
        self.device = torch.device('cuda')
        self.vgg = Vgg16(requires_grad=False)
        self.vgg = self.vgg.cuda()
        self.netR = ReliefCNN(3, 3).cuda() # relief net
        self.input_A = self.Tensor(batch_size, input_nc,
                                   fine_size, fine_size)
        self.input_B = self.Tensor(batch_size, output_nc,
                                   fine_size, fine_size)
        self.batch_size = batch_size

        # batchsize should be 1 for mask_global
        self.mask_global = torch.ByteTensor(1, 1, fine_size, fine_size)

        self.mask_global.zero_()
        self.mask_global[:, :, int(fine_size/4) + overlap: int(fine_size/2) + int(fine_size/4) - overlap,
                         int(fine_size/4) + overlap: int(fine_size/2) + int(fine_size/4) - overlap] = 1

        self.mask_type = mask_type
        self.gMask_opts = {}

        if len(self.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.cuda()

        self.netG, self.Cosis_list, self.Cosis_list2, self.CSA_model = networks.define_G(input_nc_g, output_nc, ngf,
                                                                                         which_model_netG, self.mask_global,
                                                                                         norm, use_dropout, init_type, self.gpu_ids, init_gain)
        self.netP, _, _, _ = networks.define_G(input_nc, output_nc, ngf,
                                               which_model_netP, self.mask_global,
                                               norm, use_dropout, init_type, self.gpu_ids, init_gain)
        if self.is_train:
            use_sigmoid = False
            if gan_type == 'vanilla':
                use_sigmoid = True  # only vanilla GAN using BCECriterion

            self.netD = networks.define_D(input_nc, ndf, which_model_netD,
                                          norm, use_sigmoid, init_type, self.gpu_ids, init_gain)
            self.netF = networks.define_D(input_nc, ndf, which_model_netF,
                                          norm, use_sigmoid, init_type, self.gpu_ids, init_gain)
        if not self.is_train or continue_train:
            print('Loading pre-trained network!')
            self.load_network(self.netG, 'G', which_epoch)
            self.load_network(self.netP, 'P', which_epoch)
            if self.is_train:
                self.load_network(self.netD, 'D', which_epoch)
                self.load_network(self.netF, 'F', which_epoch)

        if self.is_train:
            self.old_lr = lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(
                gan_type=gan_type, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=lr, betas=(beta1, 0.999))
            self.optimizer_P = torch.optim.Adam(self.netP.parameters(),
                                                lr=lr, betas=(beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=lr, betas=(beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(),
                                                lr=lr, betas=(beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_P)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(
                    optimizer, lr_policy, niter, niter_decay, lr_decay_iters, epoch_count))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            networks.print_network(self.netP)
            if self.is_train:
                networks.print_network(self.netD)
                networks.print_network(self.netF)
            print('-----------------------------------------------')

    def set_latent_mask(self, mask_global, layer_to_last, threshold):
        '''It is quite convinient, as one forward-pass, all the innerCos will get the GT_latent!
        '''
        self.CSA_model[0].set_mask(mask_global, layer_to_last, threshold)
        self.Cosis_list[0].set_mask(mask_global, threshold)
        self.Cosis_list2[0].set_mask(mask_global, threshold)

    def set_input(self, input, mask, mask_type='center', threshold=0.3125):
        input_A = input
        input_B = input.clone()
        input_mask = mask
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        self.image_paths = 0

        if mask_type == 'center':
            self.mask_global = self.mask_global
        elif mask_type == 'random':
            self.mask_global.zero_()
            self.mask_global = input_mask
        else:
            raise ValueError(f"Mask_type [{mask_type}] not recognized.")
            
        self.ex_mask = self.mask_global.expand(
            1, 3, self.mask_global.size(2), self.mask_global.size(3))  # 1*c*h*w
        self.inv_ex_mask = torch.add(torch.neg(self.ex_mask.float()), 1).byte()
        self.input_A.narrow(1, 0, 1).masked_fill_(
            self.mask_global.bool(), 2*123.0/255.0 - 1.0)
        self.input_A.narrow(1, 1, 1).masked_fill_(
            self.mask_global.bool(), 2*104.0/255.0 - 1.0)
        self.input_A.narrow(1, 2, 1).masked_fill_(
            self.mask_global.bool(), 2*117.0/255.0 - 1.0)
        self.set_latent_mask(self.mask_global, 3, threshold)

    def forward(self):
        self.real_A = self.input_A.to(self.device)
        # 修改
        self.fake_P = self.beta*self.real_A + (1-self.beta)*self.netR(self.real_A)
        self.fake_P = self.netP(self.fake_P)
        self.un = self.fake_P.clone()
        self.Unknowregion = self.un.data.masked_fill_(
            self.inv_ex_mask.bool(), 0)
        self.knownregion = self.real_A.data.masked_fill_(
            self.ex_mask.bool(), 0)
        self.Syn = self.Unknowregion+self.knownregion
        self.Middle = torch.cat((self.Syn, self.input_A), 1)
        self.fake_B = self.netG(self.Middle)
        self.real_B = self.input_B.to(self.device)

    def set_gt_latent(self):
        gt_latent = self.vgg(Variable(self.input_B, requires_grad=False))
        self.Cosis_list[0].set_target(gt_latent.relu4_3)
        self.Cosis_list2[0].set_target(gt_latent.relu4_3)

    def test(self):
        self.real_A = self.input_A.to(self.device)
        self.fake_P = self.netP(self.real_A)
        self.un = self.fake_P.clone()
        self.Unknowregion = self.un.data.masked_fill_(
            self.inv_ex_mask.bool(), 0)
        self.knownregion = self.real_A.data.masked_fill_(
            self.ex_mask.bool(), 0)
        self.Syn = self.Unknowregion+self.knownregion
        self.Middle = torch.cat((self.Syn, self.input_A), 1)
        self.fake_B = self.netG(self.Middle)
        self.real_B = self.input_B.to(self.device)

    def backward_D(self):
        fake_AB = self.fake_B
        # Real
        self.gt_latent_fake = self.vgg(
            Variable(self.fake_B.data, requires_grad=False))
        self.gt_latent_real = self.vgg(
            Variable(self.input_B, requires_grad=False))
        real_AB = self.real_B  # GroundTruth

        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_real = self.netD(real_AB)
        self.loss_D_fake = self.criterionGAN(
            self.pred_fake, self.pred_real, True)

        self.pred_fake_F = self.netF(self.gt_latent_fake.relu3_3.detach())
        self.pred_real_F = self.netF(self.gt_latent_real.relu3_3)
        self.loss_F_fake = self.criterionGAN(
            self.pred_fake_F, self.pred_real_F, True)

        self.loss_D = self.loss_D_fake * 0.5 + self.loss_F_fake * 0.5

        # When two losses are ready, together backward.
        # It is op, so the backward will be called from a leaf.(quite different from LuaTorch)
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = self.fake_B
        fake_f = self.gt_latent_fake

        pred_fake = self.netD(fake_AB)
        pred_fake_f = self.netF(fake_f.relu3_3)

        pred_real = self.netD(self.real_B)
        pred_real_F = self.netF(self.gt_latent_real.relu3_3)

        self.loss_G_GAN = self.criterionGAN(
            pred_fake, pred_real, False)+self.criterionGAN(pred_fake_f, pred_real_F, False)

        # Second, G(A) = B
        self.loss_G_L1 = (self.criterionL1(self.fake_B, self.real_B) +
                          self.criterionL1(self.fake_P, self.real_B)) * self.lambda_A

        self.loss_G = self.loss_G_L1 + self.loss_G_GAN * self.gan_weight

        # Third add additional netG contraint loss!
        self.ng_loss_value = 0
        self.ng_loss_value2 = 0
        if self.cosis:
            for gl in self.Cosis_list:
                #self.ng_loss_value += gl.backward()
                self.ng_loss_value += Variable(gl.loss.data,
                                               requires_grad=True)
            self.loss_G += self.ng_loss_value
            for gl in self.Cosis_list2:
                #self.ng_loss_value += gl.backward()
                self.ng_loss_value2 += Variable(gl.loss.data,
                                                requires_grad=True)
            self.loss_G += self.ng_loss_value2

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.optimizer_F.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_F.step()
        self.optimizer_G.zero_grad()
        self.optimizer_P.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_P.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                            ('G_L1', self.loss_G_L1.data.item()),
                            ('D', self.loss_D_fake.data.item()),
                            ('F', self.loss_F_fake.data.item())
                            ])

    def get_current_visuals(self):
        real_A = self.real_A.data
        fake_B = self.fake_B.data
        real_B = self.real_B.data
        return real_A, real_B, fake_B

    def save(self, epoch):
        self.save_network(self.netG, 'G', epoch)
        self.save_network(self.netP, 'P', epoch)
        self.save_network(self.netD, 'D', epoch)
        self.save_network(self.netF, 'F', epoch)
        self.save_network(self.netR, 'R', epoch)

    def load(self, epoch):
        self.load_network(self.netG, 'G', epoch)
        self.load_network(self.netP, 'P', epoch)
        if self.beta < 1:
            self.load_network(self.netP, 'R', epoch)
