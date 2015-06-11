from PIL import Image
import math
from scipy import ndimage
import numpy as np
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
from tex2label import Mapper

import sys

inkmlpath = 'test'
imgpath = 'test_expr'

# from argument
if len(sys.argv) == 3:
    inkmlpath = sys.argv[1]
    imgpath = sys.argv[2]

print "from: ", inkmlpath
print "to: ", imgpath

listfile = open(join(imgpath, 'listfile.txt'), 'w')
img_x = 28*2
img_y = 64
margin = 4*2
density = 32*2
final_width = 28
final_height = 64
mapper = Mapper('texsyms.txt')

files = [f for f in listdir(inkmlpath) if isfile(join(inkmlpath, f))]
for file_index, f in enumerate(files):
    if not f.endswith('.inkml'):
        continue
    print 'Processing', f, '...', ('(%d / %d)' % (file_index, len(files) ))

    # find traces and trace groups
    tree = ET.parse(join(inkmlpath, f))
    root = tree.getroot()
    traces = []
    traceGroup = None
    for child in root:
        if child.tag.endswith('trace'):
            traces.append(child)
        elif child.tag.endswith('traceGroup'):
            traceGroup = child

    # convert trace elements to coordinate lists
    coord = []
    for trace in traces:
        coord.append([[float(a) for a in a.split()] for a in trace.text.split(',')])

    # group traces and get symbol
    sym = []
    tex = []
    for child in traceGroup:
        if child.tag.endswith('traceGroup'):
            tmp = []
            for child in child:
                if child.tag.endswith('traceView'):
                    tmp.append(int(child.get('traceDataRef')))
                if child.tag.endswith('annotation'):
                    tex.append(child.text)
            if len(tmp) > 0:
                sym.append(tmp)

    sym = [[b for a in sym for b in a]]

    for child in root:
        if child.tag.endswith('annotation'):
            if child.get('type') == 'truth':
                gt = child.text.replace(' ', '').strip('$')

    if '^' in gt:
        continue
    if 'frac' in gt:
        continue
    if '_' in gt:
        continue

    for i in xrange(len(sym)):
        flag = False
        for a in tex:
            if mapper.tex2label(a) is None:
                flag = True
                break
        if flag:
            break
        traceref = sym[i]
        min_x = max_x = coord[sym[i][0]][0][0]
        min_y = max_y = coord[sym[i][0]][0][1]
        for j in traceref:
            for x, y in coord[j]:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        # pass one point trace
        if min_x == max_x and min_y == max_y:
            continue
        # prevent divide-by-zero
        if min_x == max_x:
            max_x = max_x + 1
        if min_y == max_y:
            max_y = max_y + 1
        img_x = int(round(img_y / (max_y - min_y) * (max_x - min_x)))
        final_width = int(round(final_height / (max_y - min_y) * (max_x - min_x)))
        x_scale = img_x / (max_x - min_x)
        y_scale = img_y / (max_y - min_y)
        if x_scale > y_scale:
            mx = (max_x + min_x) / 2
            dx = (max_x - min_x) / 2
            max_x = mx + dx / y_scale * x_scale
            min_x = mx - dx / y_scale * x_scale
        else:
            my = (max_y + min_y) / 2
            dy = (max_y - min_y) / 2
            max_y = my + dy / x_scale * y_scale
            min_y = my - dy / x_scale * y_scale

        sval=[1,2,3,5]
        for strength in xrange(0, 4):
            px = np.full((img_y, img_x), 255, dtype=np.uint8)
            for j in traceref:
                for k in xrange(1, len(coord[j])):
                    prevx = coord[j][k - 1][0]
                    prevy = coord[j][k - 1][1]
                    nextx = coord[j][k][0]
                    nexty = coord[j][k][1]
                    density = int(round(math.hypot(prevx - nextx, prevy - nexty)*2*(img_x-2*margin)/(max_x-min_x)))
                    for t in xrange(density):
                        x = prevx + (nextx - prevx) * t / density
                        y = prevy + (nexty - prevy) * t / density
                        mx = (x - min_x) / (max_x - min_x) * (img_x - 2 * margin) + margin
                        my = (y - min_y) / (max_y - min_y) * (img_y - 2 * margin) + margin
                        px[round(my), round(mx)] = 0

            img = Image.fromarray(ndimage.grey_erosion(px, size=(sval[strength],sval[strength])))
            imgname = f.rstrip('.inkml') + '_%d_%d.png' % (i, strength)
            img.resize((final_width, final_height),Image.BICUBIC).save(join(imgpath, imgname))
            listfile.write(imgname + ' ' + gt + '\n')
listfile.close()
