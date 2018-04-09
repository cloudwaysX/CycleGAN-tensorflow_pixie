#!flask/bin/python

# Author: Ngo Duy Khanh
# Email: ngokhanhit@gmail.com
# Git repository: https://github.com/ngoduykhanh/flask-file-uploader
# This work based on jQuery-File-Upload which can be found at https://github.com/blueimp/jQuery-File-Upload/

import os
import PIL
from PIL import Image
import simplejson
import traceback

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from flask_bootstrap import Bootstrap
from werkzeug import secure_filename

from lib.upload_file import uploadfile
from shutil import copyfile

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from util import util
from PIL import Image
import ntpath

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
app.config['UPLOAD_FOLDER'] = 'datasets/'
app.config['THUMBNAIL_FOLDER'] = 'datasets/thumbnail/'
app.config['STATIC_IMG_FOLDER'] = 'static/'
app.config['FOOD_FOLDER'] = 'datasets/food/'
app.config['NOSTYLE_FOLDER'] = 'datasets/style_not_imported/'
app.config['ROOM_FOLDER'] = 'datasets/room/'
app.config['PORTRAIT_FOLDER'] = 'datasets/portrait/'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['RESULT_FOLDER'] = 'datasets/result'
app.config['GPU_ID']=[0]
app.config['checkpoints_dir'] = './checkpoints'
app.config['name'] = 'vibrantandpure'
app.config['ngf'] = 64
app.config['which_model_netG'] = 'resnet_9blocks'
app.config['norm'] = 'instance'
app.config['init_type'] = 'normal'
app.config['which_epoch'] = '60'
opt = app.config

ALLOWED_EXTENSIONS = set(['txt', 'gif', 'png', 'jpg', 'jpeg', 'bmp', 'rar', 'zip', '7zip', 'doc', 'docx'])
IGNORED_FILES = set(['.gitignore'])

bootstrap = Bootstrap(app)


categories = {'food':[],'style_not_imported':[],'portrait':[],'room':[]}
categories_delete_helper = {}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def gen_file_name(filename):
    """
    If file was exist already, rename it and return a new name
    """

    i = 1
    while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
        name, extension = os.path.splitext(filename)
        filename = '%s_%s%s' % (name, str(i), extension)
        i += 1

    return filename


def create_thumbnail(image):
    try:
        base_width = 80
        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], image))
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img = img.resize((base_width, h_size), PIL.Image.ANTIALIAS)
        img.save(os.path.join(app.config['THUMBNAIL_FOLDER'], image))

        return True

    except:
        print(traceback.format_exc())
        return False


@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files = request.files['file']

        if files:
            filename = secure_filename(files.filename)
            filename = gen_file_name(filename)
            mime_type = files.content_type

            if not allowed_file(files.filename):
                result = uploadfile(name=filename, type=mime_type, size=0, not_allowed_msg="File type not allowed")

            else:
                # save file to disk
                uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
                files.save(uploaded_file_path)
                cur_category = categoryRecognition(filename)
                copyfile(os.path.join(opt['UPLOAD_FOLDER'],filename), os.path.join(opt['UPLOAD_FOLDER'], cur_category,filename))

                # create thumbnail after saving
                if mime_type.startswith('image'):
                    create_thumbnail(filename)
                
                # get file size after saving
                size = os.path.getsize(uploaded_file_path)

                # return json for js call back
                result = uploadfile(name=filename, type=mime_type, size=size)
            
            return simplejson.dumps({"files": [result.get_file()]})

    if request.method == 'GET':
        # get all file in ./datasets directory
        files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'],f)) and f not in IGNORED_FILES ]
        
        file_display = []

        for f in files:
            size = os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], f))
            file_saved = uploadfile(name=f, size=size)
            file_display.append(file_saved.get_file())

        return simplejson.dumps({"files": file_display})

    return redirect(url_for('index'))


@app.route("/delete/<string:filename>", methods=['DELETE'])
def delete(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
    server_path = url_for('send_result',filename=filename)
    # print('aa')
    # print(server_path)
    file_thumb_path = os.path.join(app.config['THUMBNAIL_FOLDER'], filename)
    result_path = os.path.join(app.config['RESULT_FOLDER'],filename)
    file_cat_path1 = os.path.join(app.config['FOOD_FOLDER'], filename)
    file_cat_path2 = os.path.join(app.config['ROOM_FOLDER'], filename)
    file_cat_path3 = os.path.join(app.config['PORTRAIT_FOLDER'], filename)
    file_cat_path4 = os.path.join(app.config['NOSTYLE_FOLDER'], filename)

    if filename in categories_delete_helper:
        categories[categories_delete_helper[filename]].remove(filename)

    if os.path.exists(file_path):
        try:
            os.remove(file_path)

            if os.path.exists(file_thumb_path):
                os.remove(file_thumb_path)
            if os.path.exists(file_cat_path1):
                os.remove(file_cat_path1)
            if os.path.exists(file_cat_path2):
                os.remove(file_cat_path2)
            if os.path.exists(file_cat_path3):
                os.remove(file_cat_path3)
            if os.path.exists(file_cat_path4):
                os.remove(file_cat_path4)
            if os.path.exists(result_path):
                os.remove(result_path)
            return simplejson.dumps({filename: 'True'})
        except:
            return simplejson.dumps({filename: 'False'})


# serve static files
@app.route("/thumbnail/<string:filename>", methods=['GET'])
def get_thumbnail(filename):
    return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename=filename)


@app.route("/datasets/<string:filename>", methods=['GET'])
def get_file(filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER']), filename=filename)

@app.route('/static/<string:filename>')
def get_static_img(filename):    
    return send_from_directory(app.config['STATIC_IMG_FOLDER'], filename=filename)

@app.route('/')
def create():
    return render_template('create.html')

@app.route('/stylize', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/style_choice',methods=['GET', 'POST'])
def get_style_choice():
    if request.method == 'POST':
        styleName_food = request.form.get('food')
        styleName_portrait = request.form.get('portrait')
        styleName_room = request.form.get('room')
        print(styleName_portrait)
        if styleName_food:
            styleTransfer(styleName_food,'latest','food')
        if styleName_portrait:
            styleTransfer(styleName_portrait,'latest','portrait')
        if styleName_room:
            styleTransfer(styleName_room,'latest','room')
        return redirect(url_for('get_gallery'))

    return render_template( 'style_choice.html', categories=categories)

def styleTransfer(styleName,degree,category):

    opt['name'] = styleName
    opt['which_epoch'] = degree
    opt['category'] = category

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

@app.route('/gallery')
def get_gallery():
    image_names = os.listdir(opt['RESULT_FOLDER'])
    return render_template( 'results.html', results=image_names)

@app.route('/results/<filename>')
def send_result(filename):    
    return send_from_directory(opt['RESULT_FOLDER'], filename)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "Pixie-1ecb2312863a.json"
def categoryRecognition(filename):
    import io
    import os

    # Imports the Google Cloud client library
    from google.cloud import vision
    from google.cloud.vision import types

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    dir = os.path.join(opt['UPLOAD_FOLDER'], filename)
    with io.open(dir, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations



    for label in labels:
        if 'food' in label.description:
            categories['food'].append(filename)
            categories_delete_helper[filename] = 'food'
            return 'food'
        if 'girl' in label.description or 'boy' in label.description or 'man' in label.description or 'woman' in label.description or 'person' in label.description or 'face' in label.description:
            categories['portrait'].append(filename)
            categories_delete_helper[filename] = 'portrait'
            return 'portrait'
        if 'room' in label.description or 'indoor' in label.description or 'house' in label.description or 'furniture' in label.description:
            categories['room'].append(filename)
            categories_delete_helper[filename] = 'room'
            return 'room'

    categories['style_not_imported'].append(filename)
    categories_delete_helper[filename] = 'style_not_imported'
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

if __name__ == '__main__':
    app.run(debug=True)
