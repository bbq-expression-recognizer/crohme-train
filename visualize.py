#-*- coding: utf-8 -*-
import numpy as np
import math
from PIL import Image
from PIL import ImageOps
from scipy import ndimage
import os
import sys

caffe_root = '../caffe/'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe


# inputs
MODEL='model.prototxt'
TRAINED='trained.caffemodel'

# caffe setting
caffe.set_mode_cpu()
net = caffe.Classifier(MODEL, TRAINED)

con1 = net.params['conv1'][0].data

tile = np.zeros((5*4,6*4,3))
tile[:,:,2] = 255

for index, fil in enumerate(con1):
    gray_array = fil[0].copy()
    gray_array = gray_array - np.min(gray_array)
    gray_array = 255 * gray_array / np.max(gray_array)
    I = Image.fromarray(np.uint8(gray_array), 'L')
    row = index / 6
    col = index - row * 6
    tile[row*4:row*4+3,col*4:col*4+3,0] = gray_array
    tile[row*4:row*4+3,col*4:col*4+3,1] = gray_array
    tile[row*4:row*4+3,col*4:col*4+3,2] = gray_array
    I.save("conv1/filter%d.png" % index)

I = Image.fromarray(np.uint8(tile))
I.save("conv1/tile.png")

con2 = net.params['conv2'][0].data

tile = np.zeros((6*4,10*4,3))
tile[:,:,2] = 255

for index, fil in enumerate(con2):
    gray_array = fil[0].copy()
    gray_array = gray_array - np.min(gray_array)
    gray_array = 255 * gray_array / np.max(gray_array)
    I = Image.fromarray(np.uint8(gray_array), 'L')
    row = index / 10
    col = index - row * 10
    tile[row*4:row*4+3,col*4:col*4+3,0] = gray_array
    tile[row*4:row*4+3,col*4:col*4+3,1] = gray_array
    tile[row*4:row*4+3,col*4:col*4+3,2] = gray_array
    I.save("conv2/filter%d.png" % index)

I = Image.fromarray(np.uint8(tile))
I.save("conv2/tile.png")

con3 = net.params['conv3'][0].data

tile = np.zeros((9*4,10*4,3))
tile[:,:,2] = 255

for index, fil in enumerate(con3):
    gray_array = fil[0].copy()
    gray_array = gray_array - np.min(gray_array)
    gray_array = 255 * gray_array / np.max(gray_array)
    I = Image.fromarray(np.uint8(gray_array), 'L')
    row = index / 10
    col = index - row * 10
    tile[row*4:row*4+3,col*4:col*4+3,0] = gray_array
    tile[row*4:row*4+3,col*4:col*4+3,1] = gray_array
    tile[row*4:row*4+3,col*4:col*4+3,2] = gray_array
    I.save("conv3/filter%d.png" % index)

I = Image.fromarray(np.uint8(tile))
I.save("conv3/tile.png")
