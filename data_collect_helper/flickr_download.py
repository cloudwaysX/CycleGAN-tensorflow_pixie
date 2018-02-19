#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 02:23:40 2018

@author: yifang
"""
import flickrapi
import urllib
api_key = '5a8d2a250ec571b58122c2453dd424a0'
api_secret = '0a5cb3ec03667757'


f=flickrapi.FlickrAPI(api_key,api_secret,cache=True)
keyward = 'iphone brunch food'
path = ''

def flickr_walk(keyward):
    count = 0
    photos = f.walk(text=keyward,
                 tag_mode='all',
                 extras='url_c',
                 per_page=5)

    for photo in photos:
        count+=1
        try:
            url=photo.get('url_c')
            urllib.request.urlretrieve(url, path+'\\' + str(count) +".jpg")
        except Exception as e:
            print('failed to download image')
        
flickr_walk(keyward)