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

def get_positions(gap, width):
    return [x*gap for x in range(width/gap)]

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

gap_ratio = 10
# caffe setting
caffe.set_mode_cpu()
net = caffe.Classifier(MODEL, TRAINED)
net_height, net_width = net.image_dims[0], net.image_dims[1]



empty_positions = np.all(binary_array == 255, axis = 0)

x_positions = [0]
for x in xrange(1, width):
    if (empty_positions[x]) and (not empty_positions[x-1]):
        x_positions.append(x)
        continue
    if (not empty_positions[x]) and (empty_positions[x-1]):
        x_positions.append(x)
        continue
x_positions.append(width)

m = len(x_positions)

# weight[i][j][d]: [i, j) 위치의 d번 symbol의 prediction value.
weights = [[[] for _ in xrange(0,m)] for _ in xrange(0,m)]

# space[i][j]: [i,j) 위치의 space prediction value. 흰색이면 1이다.
spaceval = [[0 for _ in xrange(0,m)] for _ in xrange(0,m)]

layouts = [[(0,0) for _ in xrange(0,m)] for _ in xrange(0,m)]

middle_heights = []
# fill weights
for i in xrange(0, m):
    caffe_in = np.zeros([m, 1, net_height, net_width], dtype=np.float32)
    indices = []
    for j in xrange(i+1, min(m,i+3)):
        if x_positions[j] - x_positions[i] > height:
            continue
        layout, transformed = transform_input(
            binary_array[:,x_positions[i]:x_positions[j]],
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

# fill spaces probability
for i in xrange(0, m):
    for j in xrange(i+1, m):
        x1 = x_positions[i]
        x2 = x_positions[j]
        spaceval[i][j] = pow(np.prod(np.mean(binary_array[:,x1:x2], axis=0) / 255.0), 2)

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
            for j in range(i+1, min(i+5, m)):
                newval = dp[i][depth][kind] * spaceval[i][j]
                if dp[j][depth][kind] < newval:
                    dp[j][depth][kind] = newval
                    back[j][depth][kind] = (i, depth, kind, -1)

                if len(weights[i][j]) != sigma:
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
                    newval = dp[i][depth][kind] * weights[i][j][symbol]
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


