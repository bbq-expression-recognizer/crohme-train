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

import binarization

# transforms given 2d grayscale image to height, width
def transform_input(image, height, width):
    img_x = width
    img_y = height
    margin = 4

    # crop white row padding from image
    non_white_row_indices = np.where(np.any(image < 128, axis=1))[0]
    if non_white_row_indices.size == 0:
        return None, None
    y1 = np.min(non_white_row_indices)
    y2 = np.max(non_white_row_indices)+1
    image = image[y1:y2, :]


    # crop white column padding from image
    non_white_col_indices = np.where(np.any(image < 128, axis=0))[0]
    if non_white_col_indices.size == 0:
        return None, None
    x1 = np.min(non_white_col_indices)
    x2 = np.max(non_white_col_indices)+1
    image = image[:, x1:x2]

    resized = Image.fromarray(image,mode='L')
    resized.thumbnail((img_x-2*margin, img_y-2*margin),Image.ANTIALIAS)
    result = np.full([img_y,img_x], 255.0, dtype=np.float32)
    resized_size = resized.size

    rowbeg = margin
    rowend = height-margin
    if rowend-rowbeg > resized_size[1]:
        dec = rowend-rowbeg-resized_size[1]
        rowbeg = rowbeg + dec/2
        rowend = rowend - (dec - dec/2)

    colbeg = margin
    colend = width-margin
    if colend-colbeg > resized_size[0]:
        dec = colend-colbeg-resized_size[0]
        colbeg = colbeg + dec/2
        colend = colend - (dec - dec/2)

    result[rowbeg:rowend, colbeg:colend] = resized
    result[result<200] = 0
    return (y1, y2, x1, x2), result/255.0

# inputs
MODEL='model.prototxt'
TRAINED='trained.caffemodel'

if len(sys.argv) != 2:
    print ("Usage: python main.py (image-file-name)")
    exit(1)

IMAGE=sys.argv[1]

if not os.path.isfile(TRAINED):
    print ("file not found: %s" % TRAINED)
    exit(1)
if not os.path.isfile(MODEL):
    print ("file not found: %s" % MODEL)
    exit(1)

# caffe setting
caffe.set_mode_cpu()
net = caffe.Classifier(MODEL, TRAINED)
net_height, net_width = net.image_dims[0], net.image_dims[1]

# Overall pipeline
# 1. convert image to binary image.
# 2. get best rotation angle (which makes minimum height deviation) and rotate.
# 3. divide images with vertical lines
# 4. classify each region
# 5. use Dynamic Programming approach to get the most likely formula.
# 6. back track dynamic programming table.


#PIL image
gray_image = Image.open(IMAGE).convert('L')

#numpy 2d array (0~255)
binary_array = binarization.to_binary(gray_image)

# rotate to horizontal line
rotated_angle = binarization.rotated_angle(binary_array)
binary_array = binarization.rotate_image(binary_array, rotated_angle)
# Image.fromarray(binary_array, 'L').save('rotate_normalized.png')

height, width = binary_array.shape

# labeling with flood-fill (8-direction)
labeled_array, num_features = ndimage.measurements.label(np.invert(binary_array),
        structure = ndimage.morphology.generate_binary_structure(2, 2))
features = []
for i in xrange(1, num_features + 1):
    x_coords = (labeled_array == i).nonzero()[1]
    xmin = min(x_coords)
    xmax = max(x_coords)
    # (start x coordinate, end x coordinate, feature number)
    features.append([min(x_coords), max(x_coords), i])
# sort by starting coordinate
features.sort()

if len(features) == 0:
    print 'no symbols detected'
    exit(1)

# virtual symbol at start (for easy dp)
merged_features = [[-1, -1, 0]]
for i in xrange(num_features):
    # no overlapping
    if merged_features[-1][1] < features[i][0]:
        merged_features.append(features[i])
    # overlap
    else:
        overlap = float(min(merged_features[-1][1], features[i][1]) - features[i][0])
        ratio0 = overlap / (merged_features[-1][1] - merged_features[-1][0])
        ratio1 = overlap / (features[i][1] - features[i][0])
        # > 50% overlap from both features' perspective
        if ratio0 > 0.5 or ratio1 > 0.5:
            # merge coords
            merged_features[-1][1] = max(merged_features[-1][1], features[i][1])
            # change label
            labeled_array[labeled_array == features[i][2]] = merged_features[-1][2]
        else:
            merged_features.append(features[i])

# swap to merged one
features = merged_features
m = len(features)
#print features

# weight[i][j][d]: [i, j] 위치의 d번 symbol의 prediction value.
weights = [[[] for _ in xrange(0, m)] for _ in xrange(0, m)]

# space[i][j]: [i,j] 위치의 space prediction value. 흰색이면 1이다.
spaceval = [[0 for _ in xrange(0, m)] for _ in xrange(0, m)]

layouts = [[(0,0) for _ in xrange(0,m)] for _ in xrange(0,m)]

middle_heights = []
# fill weights
for i in xrange(1, m):
    caffe_in = np.zeros([m, 1, net_height, net_width], dtype=np.float32)
    indices = []
    # we need image buffer, since we cannot directly cut image due to overlap
    imgbuf = np.full([height, width], 255, dtype = np.uint8)
    for j in xrange(i, min(i + 1, m)):
        # too long, doesn't seem to be symbol
        if features[j][1] - features[i][0] > height:
            continue
        # fill current feature
        imgbuf[labeled_array == features[j][2]] = 0
        layout, transformed = transform_input(
            imgbuf[:, features[i][0] : features[j][1] + 1],
            net_height,
            net_width)
        if transformed is None:
            continue
        layouts[i][j] = layout
        middle_heights.append((layout[0] + layout[1]-1)/2)
        caffe_in[len(indices)][0] = transformed
        indices.append(j)
        #img = Image.fromarray(transformed*255,mode='F')
        #img = img.convert('L')
        #img.save('test%d-%d.png'%(i,j))
    out = net.forward_all(**{net.inputs[0]: caffe_in[0:len(indices)]})
    predictions = out[net.outputs[0]]

    for j, pred in zip(indices,predictions):
        weights[i][j] = pred

middle_height = np.mean(middle_heights)

# calculate bottom line and top line
black_cells = np.where(binary_array == 0)[0]
top = int(np.percentile(black_cells, 1))
bottom = int(np.percentile(black_cells, 99))

# fill spaces probability
for i in xrange(1, m):
    imgbuf = np.full([height, width], 255, dtype = np.uint8)
    for j in xrange(i, m):
        imgbuf[labeled_array == features[j][2]] = 0
        spaceval[i][j] = pow(np.prod(np.mean(imgbuf[top:bottom+1, features[i][0] : features[j][1] + 1], axis=0) / 255.0), 2)

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
sym2kind=[ 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 2, 2, 2, 0]


sigma = len(syms)

open_symbol={0,16,20}
closing_symbol={1,22,21}
digits_symbol={5,6,7,8,9,10,11,12,13,14,18,23}
operator_symbol={2,3,4,15,17,19}

# check if syms[s1] to syms[s2] is possible
def possible_transition(s1,s2):
    if (syms[s1] == "\\pi") and (s2 >= 5 and s2 <= 14):
        return False
    if (syms[s1] == "e") and (s2 >= 5 and s2 <= 14):
        return False
    if (s1 in operator_symbol) and (s2 in operator_symbol):
        return False
    if (s1 in operator_symbol) and (s2 in closing_symbol):
        return False
    if (s1 in open_symbol) and (s2 in operator_symbol) and (syms[s2] != "-"):
        return False
    return True

# dp[i][open][kind]: i번째 위치까지. open된 parenthesis 개수, 마지막 symbol 번호
dp = [[[0 for _ in xrange(0, sigma)] for _ in xrange(0,m)] for _ in xrange(0,m)]
# for backtracking
back = [[[(-1, -1, -1, -1) for _ in xrange(0, sigma)] for _ in xrange(0,m)] for _ in xrange(0,m)]

for x in range(0, sigma):
    dp[0][0][x] = 1

for i in range(0,m):
    for depth in range(0,i+1):
        for kind in range(0, sigma):
            if dp[i][depth][kind] == 0:
                continue
            for j in range(i+1, min(i+3, m)):
                newval = dp[i][depth][kind] * spaceval[i + 1][j]
                if dp[j][depth][kind] < newval:
                    dp[j][depth][kind] = newval
                    back[j][depth][kind] = (i, depth, kind, -1)

                if len(weights[i + 1][j]) != sigma:
                    continue
                for symbol in range(0,sigma):
                    if symbol in open_symbol:
                        next_depth = depth + 1
                    elif symbol in closing_symbol:
                        next_depth = depth - 1
                        if next_depth < 0:
                            continue
                    else:
                        next_depth = depth

                    if not possible_transition(kind, symbol):
                        continue
                    newval = dp[i][depth][kind] * weights[i+1][j][symbol]
                    if dp[j][next_depth][symbol] < newval:
                        dp[j][next_depth][symbol] = newval
                        back[j][next_depth][symbol] = (i, depth, kind, symbol)

def follow_dp():
    i = m-1
    depth = 0
    kind = np.argmax(dp[m-1][0])

    res = []

    while i >= 0:
        i, depth, kind, symbol = back[i][depth][kind]
        if symbol >= 0:
            if symbol in open_symbol:
                res.append('(')
            elif symbol in closing_symbol:
                res.append(')')
            else:
                res.append(syms[symbol])

    return "".join(res[::-1])

print follow_dp()
