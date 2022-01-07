import numpy as np

from torch import ByteTensor
from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks






class TestModel(BaseModel):
    def __init__(self, fine_size, name, checkpoints_dir, gpu_ids,
                 batch_size, input_nc, input_nc_g, output_nc, ngf,
                 which_model_netG, norm, use_dropout,
                 init_type, init_gain):
        '''
        kw = (fine_size, init_gain, input_nc, input_nc_g, output_nc, norm, use_dropout, init_type)
        '''
        super().__init__(name, False, checkpoints_dir, gpu_ids)
        self.mask_global = ByteTensor(1, 1, fine_size, fine_size)
        self.input_A = self.Tensor(batch_size, input_nc,
                                   fine_size, fine_size)
        self.netG = networks.define_G(input_nc_g, output_nc, ngf,
                                      which_model_netG, self.mask_global,
                                      norm, use_dropout, init_type, self.gpu_ids, init_gain)
        networks.print_network(self.netG)

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
