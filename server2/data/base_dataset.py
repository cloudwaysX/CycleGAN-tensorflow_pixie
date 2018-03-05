import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

LOAD_SIZE = 286
FINE_SIZE = 256

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []

    osize = [LOAD_SIZE, LOAD_SIZE]
    transform_list.append(transforms.Scale(osize, Image.BICUBIC))
#        transform_list.append(transforms.RandomCrop(opt.fineSize))
    transform_list.append(transforms.CenterCrop(FINE_SIZE))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
