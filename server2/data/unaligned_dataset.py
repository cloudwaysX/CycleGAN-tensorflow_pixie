import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = 'datasets'
#        self.dir_A = os.path.join(opt.dataroot, 'testA')
        self.dir_A = os.path.join(opt['UPLOAD_FOLDER'])

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)

        return {'A': A,
                'A_paths': A_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'UnalignedDataset'
    
    import os
