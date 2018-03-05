import sys
import os
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from PIL import Image
import sys

#from options.test_options import TestOptions
#from data.data_loader import CreateDataLoader
#from models.models import create_model
#from util.visualizer import Visualizer
#from util import html
#
#opt = TestOptions().parse()
#opt.nThreads = 1   # test code only supports nThreads = 1
#opt.batchSize = 1  # test code only supports batchSize = 1
#opt.serial_batches = True  # no shuffle
#opt.no_flip = True  # no flip
#
#data_loader = CreateDataLoader(opt)
#dataset = data_loader.load_data()
#model = create_model(opt)

# =============================================================================
# 
# =============================================================================



from flask import Flask, render_template, request,redirect, url_for,flash,send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)


UPLOAD_FOLDER = 'datasets/testA'
RESULT_FOLDER = 'datasets/fakeB'
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg'])




import ntpath
from scipy.misc import imresize
from util import util

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['GPU_ID']=[0]
app.config['checkpoints_dir'] = './checkpoints'
app.config['name'] = 'flickrFood2InsFood'
app.config['ngf'] = 64
app.config['which_model_netG'] = 'resnet_9blocks'
app.config['norm'] = 'instance'
app.config['init_type'] = 'normal'
app.config['which_epoch'] = '40'

opt = app.config




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
#        for f in os.listdir(app.config['UPLOAD_FOLDER']):
#            os.remove(f)        
#        for f in os.listdir(app.config['RESULT_FOLDER']):
#            os.remove(f) 
#        
#        
        # check if the post request has the file part
        if 'file[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist("file[]")
        # if user does not select file, browser also
        # submit a empty part without filename
        if files[0].filename == '':
            flash('No selected file')
            return redirect(request.url)
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        styleTransfer()
        return redirect(url_for('get_gallery'))
            
    return render_template('uploadImg.html')


@app.route('/uploads/<filename>')
def send_image(filename):    
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/gallery')
def get_gallery():
    image_names = os.listdir(app.config['RESULT_FOLDER'])
    return render_template( 'results.html', results=image_names)
 
    
def styleTransfer():
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    for i, data in enumerate(dataset):
    #    if i >= opt.how_many:
    #        break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        save_images(app.config['RESULT_FOLDER'],visuals,img_path)
           
    

def save_images(image_dir, visuals, image_path, aspect_ratio=1.0):
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    im = visuals
    image_name = '%s.jpg' % (name)
    save_path = os.path.join(image_dir, image_name)
    h, w, _ = im.shape
    if aspect_ratio > 1.0:
        im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
    if aspect_ratio < 1.0:
        im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
    util.save_image(im, save_path)

# test

    



