#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:32:33 2018

@author: yifang
"""

import torch.utils.data as data

from PIL import Image
from PIL import ImageEnhance
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def data_aug(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img = Image.open(path,'r')
                for i in range(4):
                    img2 = img.rotate(90*i)
                    img2.save(os.path.join(color_folder,str(i)+'_'+fname))
                img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
                img2.save(os.path.join(color_folder,'_f1'+fname))
                img2 = img.transpose(Image.FLIP_TOP_BOTTOM)
                img2.save(os.path.join(color_folder,'_f2'+fname))                

    return images

import numpy as np
import colorsys

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = h+hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')
    

    return new_img.convert("RGB")

def make_BW(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                BW_img = Image.open(path,'r')
                BW_img = BW_img.convert('L')
                BW_img.save(os.path.join(BW_folder,fname))

    return images

def make_badImg(dir):
    from random import uniform
    
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                bad_img = Image.open(path,'r')
                converter1 = ImageEnhance.Color(bad_img)
                bad_img2 = converter1.enhance(uniform(0.5,0.9))
                converter2 = ImageEnhance.Contrast(bad_img2)
                bad_img3 = converter2.enhance(uniform(0.5,1.0)) 
                converter3 = ImageEnhance.Brightness(bad_img3)
                bad_img4 = converter3.enhance(uniform(0.7,1.3)) 
                converter4 = ImageEnhance.Sharpness(bad_img4)
                bad_img5 = converter4.enhance(uniform(0.8,1.0))
                if uniform(0,1)<0.5:
                    bad_img5 = colorize(bad_img5,uniform(-25,-12))
                else:
                    bad_img5 = colorize(bad_img5,uniform(12,25))
                bad_img5.save(os.path.join(bad_folder,fname))

    return images



BW_folder = '/home/yifang/pytorch-CycleGAN-and-pix2pix/datasets/food/A/train'
bad_folder = BW_folder
if not os.path.exists(BW_folder):
    os.makedirs(BW_folder)

color_folder = '/home/yifang/pytorch-CycleGAN-and-pix2pix/datasets/food/B/train'
#data_aug('/home/yifang/food_pics/train')
#make_BW(color_folder)
make_badImg(color_folder)

#BW_folder = '/home/yifang/pytorch-CycleGAN-and-pix2pix/datasets/food/A/test'
#bad_folder = BW_folder
#if not os.path.exists(BW_folder):
#    os.makedirs(BW_folder)
#
#color_folder = '/home/yifang/pytorch-CycleGAN-and-pix2pix/datasets/food/B/test'
##data_aug('/home/yifang/food_pics/test')
##make_BW(color_folder)
#make_badImg(color_folder)