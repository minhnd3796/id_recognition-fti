from os import listdir, makedirs
from os.path import join, exists
import caffe
import numpy as np
from skimage import color
import skimage
import sys
from shutil import copyfile

def OCR_digit_recognise(filename, net):
    img = caffe.io.load_image(filename)
    img_gray = color.rgb2gray(img)
    # img_gray = 1 - img_gray # invert the white background image
    img_gray_resized = skimage.transform.resize(img_gray, (28, 28), mode='constant')
    net.blobs['data'].data[...] = img_gray_resized[np.newaxis, np.newaxis, :, :]
    out = net.forward()
    result = out['prob'].argmax()
    return result

if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    _net = caffe.Net('lenet_deploy.prototxt', 'mixed_lenet_8817_id_digit.caffemodel', caffe.TEST)

    input_dir = 'cropped'
    output_dir = 'labelled'

    files = listdir(input_dir)
    num_files = len(files)
    i = 0
    for file in files:
        i += 1
        result = OCR_digit_recognise(join(input_dir, file), _net)
        result_dir = join(output_dir, str(result))
        if not exists(result_dir):
            makedirs(result_dir)
        copyfile(join(input_dir, file), join(result_dir, file))
        print('Classifying file # ' + str(i) + "/" + str(num_files) + ' done!')