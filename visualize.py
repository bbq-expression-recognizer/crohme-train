#-*- coding: utf-8 -*-
import numpy as np
import math
from PIL import Image
from PIL import ImageOps
from scipy import ndimage
import os
import sys
import shutil

caffe_root = '../caffe/'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe


# inputs
MODEL='model.prototxt'
TRAINED='trained.caffemodel'

# caffe setting
caffe.set_mode_cpu()
net = caffe.Classifier(MODEL, TRAINED)

rows = [5,6,9]
cols = [6,10,10]

if True:
    net_height, net_width = net.image_dims[0], net.image_dims[1]
    PATH = '../project/inkml-test-png/'
    test_list = open(PATH + 'listfile.txt','r')
    l = [x.split() for x in test_list.readlines()]
    test_list.close()
    l = l[::1000]

    confusion = [[0 for _ in xrange(0, 24)] for _ in xrange(0,24)]
    caffe_in = np.zeros([len(l), 1, net_height, net_width], dtype=np.float32)
    for ind, (filename, symbol_id) in enumerate(l):
        print "processing %d/%d image (%s)" % (ind, len(l), filename)
        gray_image = Image.open(PATH + filename).convert('L')
        caffe_in[ind][0] = np.asarray(gray_image).copy()
    out = net.forward_all(**{net.inputs[0]: caffe_in})

    for ind, (filename, symbol_id) in enumerate(l):
        try:
            os.mkdir("blob%d" % ind)
        except:
            pass
        shutil.copy2(PATH + filename, "blob%d/original.png" % ind)

        cellwidths = [27, 12, 5]
        for level in xrange(0,3):
            w = cellwidths[level] # cell size
            tile = np.zeros((rows[level]*w,cols[level]*w,3))
            tile[:,:,2] = 255
            convstr = 'conv%d' % (level + 1)
            for iind, img in enumerate(net.blobs[convstr].data[0, :]):
                grayscale = img.copy()
                grayscale = grayscale - np.min(grayscale)
                grayscale = 255 * grayscale / np.max(grayscale)
                I = Image.fromarray(np.uint8(grayscale), 'L')
                I.save('blob%d/%s-%d.png' % (ind, convstr, iind))
                row = iind / cols[level]
                col = iind - row * cols[level]
                tile[row*w:row*w+w-1,col*w:col*w+w-1,0] = grayscale
                tile[row*w:row*w+w-1,col*w:col*w+w-1,1] = grayscale
                tile[row*w:row*w+w-1,col*w:col*w+w-1,2] = grayscale
            I = Image.fromarray(np.uint8(tile))
            I.save('blob%d/%s-tile.png' % (ind, convstr))


con1 = net.params['conv1'][0].data

w = 4 # cell size
tile = np.zeros((rows[0]*w,cols[0]*w,3))
tile[:,:,2] = 255

for index, fil in enumerate(con1):
    gray_array = fil[0].copy()
    gray_array = gray_array - np.min(gray_array)
    gray_array = 255 * gray_array / np.max(gray_array)
    I = Image.fromarray(np.uint8(gray_array), 'L')
    row = index / cols[0]
    col = index - row * cols[0]
    tile[row*w:row*w+w-1,col*w:col*w+w-1,0] = gray_array
    tile[row*w:row*w+w-1,col*w:col*w+w-1,1] = gray_array
    tile[row*w:row*w+w-1,col*w:col*w+w-1,2] = gray_array
    I.save("conv1/filter%d.png" % index)

I = Image.fromarray(np.uint8(tile))
I.save("conv1/tile.png")

con2 = net.params['conv2'][0].data

w = 4 # cell size
tile = np.zeros((rows[1]*w,cols[1]*w,3))
tile[:,:,2] = 255

for index, fil in enumerate(con2):
    gray_array = fil[0].copy()
    gray_array = gray_array - np.min(gray_array)
    gray_array = 255 * gray_array / np.max(gray_array)
    I = Image.fromarray(np.uint8(gray_array), 'L')
    row = index / cols[1]
    col = index - row * cols[1]
    tile[row*w:row*w+w-1,col*w:col*w+w-1,0] = gray_array
    tile[row*w:row*w+w-1,col*w:col*w+w-1,1] = gray_array
    tile[row*w:row*w+w-1,col*w:col*w+w-1,2] = gray_array
    I.save("conv2/filter%d.png" % index)

I = Image.fromarray(np.uint8(tile))
I.save("conv2/tile.png")

con3 = net.params['conv3'][0].data

w = 4 # cell size
tile = np.zeros((rows[2]*w,cols[2]*w,3))
tile[:,:,2] = 255

for index, fil in enumerate(con3):
    gray_array = fil[0].copy()
    gray_array = gray_array - np.min(gray_array)
    gray_array = 255 * gray_array / np.max(gray_array)
    I = Image.fromarray(np.uint8(gray_array), 'L')
    row = index / cols[2]
    col = index - row * cols[2]
    tile[row*w:row*w+w-1,col*w:col*w+w-1,0] = gray_array
    tile[row*w:row*w+w-1,col*w:col*w+w-1,1] = gray_array
    tile[row*w:row*w+w-1,col*w:col*w+w-1,2] = gray_array
    I.save("conv3/filter%d.png" % index)

I = Image.fromarray(np.uint8(tile))
I.save("conv3/tile.png")
