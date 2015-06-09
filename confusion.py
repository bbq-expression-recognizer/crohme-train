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

def parse_expression(net):
    net_height, net_width = net.image_dims[0], net.image_dims[1]

    PATH = '../project/inkml-test-png/'
    test_list = open(PATH + 'listfile.txt','r')
    l = [x.split() for x in test_list.readlines()]
    test_list.close()

    confusion = [[0 for _ in xrange(0, 24)] for _ in xrange(0,24)]
    caffe_in = np.zeros([len(l), 1, net_height, net_width], dtype=np.float32)
    for ind, (filename, symbol_id) in enumerate(l):
        print "processing %d/%d image (%s)" % (ind, len(l), filename)
        gray_image = Image.open(PATH + filename).convert('L')
        caffe_in[ind][0] = np.asarray(gray_image).copy()
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    predictions = out[net.outputs[0]]

    for ind, (filename, symbol_id) in enumerate(l):
        real = int(symbol_id)
        predicted = np.argmax(predictions[ind])
        confusion[real][predicted] = confusion[real][predicted] + 1

    #0 (
    #1 )
    #2 +
    #3 -
    #4 /
    #5 0
    #6 1
    #7 2
    #8 3
    #9 4
    #10 5
    #11 6
    #12 7
    #13 8
    #14 9
    #15 =
    #16 [
    #17 \div
    #18 \pi
    #19 \times
    #20 \{
    #21 \}
    #22 ]
    #23 e

    syms=[ "(", ")", "+", "-", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "[", "\\div", "\\pi", "\\times", "\\{", "\\}", "]", "e"]

    for actual in xrange(0,24):
        samples = np.sum(confusion[actual])
        for predicted in xrange(0,24):
            print " {:7.2%} ".format(confusion[actual][predicted] / float(samples)),
        print (" #%d" % samples)

    return confusion


# inputs
MODEL='model.prototxt'
TRAINED='trained.caffemodel'

# caffe setting
caffe.set_mode_cpu()
net = caffe.Classifier(MODEL, TRAINED)

print parse_expression(net)

