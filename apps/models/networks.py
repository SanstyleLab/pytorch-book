from torch import nn, cat
from torch.nn import functional as F
from torch.optim import lr_scheduler

from .InnerCos import InnerCos
from .InnerCos2 import InnerCos2
from .CSA_model import CSA_model
from .utils import get_norm_layer, init_net, print_network
from .utils import UnetSkipConnectionBlock_3, UnetGenerator, NLayerDiscriminator, PFDiscriminator, GANLoss


def get_scheduler(optimizer, lr_policy, niter, niter_decay, lr_decay_iters, epoch_count):
    if lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + epoch_count -
                             niter) / float(niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=niter,
                                                   eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


def define_D(input_nc, ndf, which_model_netD,
             norm='batch', use_sigmoid=False,
             init_type='normal', gpu_ids=[], init_gain=0.02):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(
            input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'feature':
        netD = PFDiscriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)


class CSA(nn.Module):
    def __init__(self, outer_nc, inner_nc, csa_model, cosis_list, cosis_list2, mask_global, input_nc,
                 submodule=None,  outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, threshold=0.3125, strength=1, fixed_mask=1, shift_sz=1, stride=1, mask_thred=1, triple_weight=1, skip=0):
        super().__init__()
        '''
        shift_sz: size of feature patch
        '''
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv_3 = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                               stride=1, padding=1)
        downrelu_3 = nn.LeakyReLU(0.2, True)
        downnorm_3 = norm_layer(inner_nc, affine=True)
        uprelu_3 = nn.ReLU(True)
        upnorm_3 = norm_layer(outer_nc, affine=True)

        downconv = nn.Conv2d(input_nc, input_nc, kernel_size=4,
                             stride=2, padding=3, dilation=2)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(input_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)
        csa = CSA_model(threshold, fixed_mask, shift_sz,
                        stride, mask_thred, triple_weight)
        csa.set_mask(mask_global, 3, threshold)
        csa_model.append(csa)
        innerCos = InnerCos(strength=strength, skip=skip)
        # Here we need to set mask for innerCos layer too.
        innerCos.set_mask(mask_global, threshold)
        cosis_list.append(innerCos)

        innerCos2 = InnerCos2(strength=strength, skip=skip)
        # Here we need to set mask for innerCos layer too.
        innerCos2.set_mask(mask_global, threshold)
        cosis_list2.append(innerCos2)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv_3 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                          kernel_size=3, stride=1,
                                          padding=1)
            down = [downconv_3]
            up = [uprelu, upconv_3]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # for the innermost, no submodule, and delete the bn
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            upconv = nn.ConvTranspose2d(outer_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            upconv_3 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                          kernel_size=3, stride=1,
                                          padding=1)
            down = [downrelu, downconv, downnorm, downrelu_3,
                    downconv_3, csa, innerCos, downnorm_3]
            up = [innerCos2, uprelu_3, upconv_3,
                  upnorm_3, uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:  # if it is the outermost, directly pass the input in.
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.upsample(x_latter, (h, w), mode='bilinear')
            return cat([x_latter, x], 1)  # cat in the C channel


class UnetGeneratorCSA(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, mask_global, csa_model, cosis_list, cosis_list2, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock_3(ngf * 8, ngf * 8,
                                               input_nc=None, submodule=None,
                                               norm_layer=norm_layer, innermost=True)

        # The innner layers number is 3 (sptial size:512*512), if unet_256.
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock_3(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock_3(
            ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_csa = CSA(ngf * 4, ngf * 8, csa_model, cosis_list, cosis_list2,
                       mask_global, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_3(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_csa, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_3(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_3(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


def define_G(input_nc, output_nc, ngf,
             which_model_netG, mask_global,
             norm='batch', use_dropout=False, init_type='normal',
             gpu_ids=[], init_gain=0.02):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    cosis_list = []
    cosis_list2 = []
    csa_model = []
    if which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf,
                             norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_csa':
        netG = UnetGeneratorCSA(input_nc, output_nc, 8, mask_global, csa_model,
                                cosis_list, cosis_list2, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % which_model_netG)

    return init_net(netG, init_type, init_gain, gpu_ids), cosis_list, cosis_list2, csa_model
