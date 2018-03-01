import caffe
import numpy as np
from skimage import color
import skimage
import imageio
import matplotlib.pyplot as plt
from os.path import exists, join
from os import listdir, makedirs
from shutil import copyfile
import sys
import cv2
from math import floor

def square_padding(dir_name, filename):
    ori_img = cv2.imread(join(dir_name, filename), 0)
    squared_img = ori_img
    height, width = ori_img.shape
    if (width != height):
        BLACK = [0, 0, 0]
        if (width > height):
            diff = width - height
            if (diff % 2 == 0):
                padding = int(diff / 2)
                squared_img = cv2.copyMakeBorder(ori_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT,value=BLACK)
            else:
                padding = floor(diff / 2)
                squared_img = cv2.copyMakeBorder(ori_img, padding+1, padding, 0,0, cv2.BORDER_CONSTANT,value=BLACK)
        else:
            diff = height - width
            if (diff % 2 == 0):
                padding = int(diff / 2)
                squared_img = cv2.copyMakeBorder(ori_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT,value=BLACK)
            else:
                padding = floor(diff / 2)
                squared_img = cv2.copyMakeBorder(ori_img, 0,0, padding+1, padding, cv2.BORDER_CONSTANT,value=BLACK)
    return squared_img

def OCR_digit_recognise(filename, net):
    img = caffe.io.load_image(filename)
    img_gray = color.rgb2gray(img)
    # img_gray = 1 - img_gray # invert the white background image
    img_gray_resized = skimage.transform.resize(img_gray, (28, 28), mode='constant')
    net.blobs['data'].data[...] = img_gray_resized[np.newaxis, np.newaxis, :, :]
    out = net.forward()
    result = out['prob'].argmax()
    return result

# caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()
_net = caffe.Net('lenet_deploy.prototxt', 'mixed_lenet_8817_id_digit.caffemodel', caffe.TEST)

dir_name = join('labeling_new', sys.argv[1])
files = listdir(dir_name)
if not exists(dir_name + '_padded'):
    makedirs(dir_name + '_padded')
for file in files:
    cv2.imwrite(dir_name + '_padded/' + file, square_padding(dir_name, file))

src_dir = dir_name + '_padded'
digits = listdir(src_dir)
if not exists(src_dir + '_labeled'):
    makedirs(src_dir + '_labeled')
for digit in digits:
    result = OCR_digit_recognise(join(src_dir, digit), _net)
    result_dir = join(src_dir + '_labeled', str(result))
    # print(result_dir)
    if not exists(result_dir):
        makedirs(result_dir)
    copyfile(join(src_dir, digit), join(result_dir, digit))