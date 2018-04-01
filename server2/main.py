import sys
import os
import os
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from PIL import Image
import sys
from shutil import copyfile
# import SR


# =============================================================================
# 
# =============================================================================



from flask import Flask, render_template, request,redirect, url_for,flash,send_from_directory
from werkzeug.utils import secure_filename


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "Pixie-1ecb2312863a.json"

app = Flask(__name__)


UPLOAD_FOLDER = 'datasets/testA'
RESULT_FOLDER = 'datasets/fakeB'
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg'])
categories = {'food':[],'style_not_imported':[]}



import ntpath
from scipy.misc import imresize
from util import util

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['GPU_ID']=[0]
app.config['checkpoints_dir'] = './checkpoints'
app.config['name'] = 'vibrantandpure'
app.config['ngf'] = 64
app.config['which_model_netG'] = 'resnet_9blocks'
app.config['norm'] = 'instance'
app.config['init_type'] = 'normal'
app.config['which_epoch'] = '60'

opt = app.config



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        cleanFolder()
       
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
                file.save(os.path.join(opt['UPLOAD_FOLDER'],'unlabeled',filename))
                cur_category = categoryRecognition(filename)
                copyfile(os.path.join(opt['UPLOAD_FOLDER'],'unlabeled',filename), os.path.join(opt['UPLOAD_FOLDER'], cur_category,filename))
        return redirect(url_for('get_style_choice'))
            
    return render_template('uploadImg.html')

@app.route('/uploads/<filename>')
def send_upload(filename):    
    print(filename)
    return send_from_directory(opt['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def send_result(filename):    
    return send_from_directory(opt['RESULT_FOLDER'], filename)

@app.route('/style_choice',methods=['GET', 'POST'])
def get_style_choice():
    if request.method == 'POST':
        app.config['name'] = 'flickrFood2InsFood'
        styleName = request.form.get('style_select')
        degree = str(request.form.get('degree_select'))
        styleTransfer(styleName,str(degree))
        return redirect(url_for('get_gallery'))

    image_names = os.listdir(opt['UPLOAD_FOLDER'])
    return render_template( 'style_choice.html', categories=categories)

@app.route('/gallery')
def get_gallery():
    image_names = os.listdir(opt['RESULT_FOLDER'])
    return render_template( 'results.html', results=image_names)
 
    
def styleTransfer(styleName,degree):

    opt['name'] = styleName
    opt['which_epoch'] = degree
    opt['category'] = 'food'

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
        save_images(opt['RESULT_FOLDER'],visuals,img_path)

def categoryRecognition(filename):
    import io
    import os

    # Imports the Google Cloud client library
    from google.cloud import vision
    from google.cloud.vision import types

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    dir = os.path.join(opt['UPLOAD_FOLDER'], 'unlabeled',filename)
    with io.open(dir, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    for label in labels:
        if 'food' in label.description:
            categories['food'].append(filename)
            return 'food'
    categories['style_not_imported'].append(filename)
    return 'style_not_imported'

           
    

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

def cleanFolder():
    temp_dict = {}
    dirPath = opt['RESULT_FOLDER']
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath+"/"+fileName)
    dirPath = opt['UPLOAD_FOLDER']
    subFolderList = os.listdir(dirPath)
    for subFolder in subFolderList:
        for fileName in os.listdir(os.path.join(dirPath,subFolder)):
            os.remove(os.path.join(dirPath,subFolder,fileName))


