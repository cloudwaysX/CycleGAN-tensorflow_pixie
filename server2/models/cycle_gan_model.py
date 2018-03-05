from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks

class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(3, 3,
                                        opt['ngf'], opt['which_model_netG'], opt['norm'], False, opt['init_type'], self.gpu_ids)
        which_epoch = opt['which_epoch']
        self.load_network(self.netG_A, 'G_A', which_epoch)



    def set_input(self, input):
        input_A = input['A']
        
        
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)

        self.input_A = input_A
        self.image_paths = input['A_paths']
        


    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        fake_B = self.netG_A(real_A)
        self.fake_B = fake_B.data


    # get image paths
    def get_image_paths(self):
        return self.image_paths
    

    def get_current_visuals(self):
        fake_B = util.tensor2im(self.fake_B)
        return fake_B
