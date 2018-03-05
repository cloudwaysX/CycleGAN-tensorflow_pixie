import os
from data.data_loader import CreateDataLoader
from models.models import create_model
import ntpath
from scipy.misc import imresize
from util import util

opt = {'UPLOAD_FOLDER':'datasets','GPU_ID':[0]}
opt['checkpoints_dir'] = './checkpoints'
opt['name'] = 'flickrFood2InsFood'
opt['ngf'] = 64
opt['which_model_netG'] = 'resnet_9blocks'
opt['norm'] = 'instance'
opt['init_type'] = 'normal'
opt['which_epoch'] = '40'

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)



def save_images(image_dir, visuals, image_path, aspect_ratio=1.0):
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    im = visuals
    image_name = '%s.png' % (name)
    save_path = os.path.join(image_dir, image_name)
    h, w, _ = im.shape
    if aspect_ratio > 1.0:
        im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
    if aspect_ratio < 1.0:
        im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
    util.save_image(im, save_path)

# test
for i, data in enumerate(dataset):
#    if i >= opt.how_many:
#        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    save_images('datasets/fakeB',visuals,img_path)
    

#