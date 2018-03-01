import numpy as np
import caffe
from skimage import color
from labeled import count_files
from os.path import join
from os import listdir
import pandas as pd

def cal_conf(filename, net):
    img = caffe.io.load_image(filename)
    img_gray = color.rgb2gray(img)
    # img_gray_resized = skimage.transform.resize(img_gray, (28, 28), mode='constant')

    net.blobs['data'].data[...] = img_gray[np.newaxis, np.newaxis, :, :]

    out = net.forward()
    return out['prob'].max(), out['prob'].argmax()

# caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

num_files = count_files.count_files('labeled/original_size')
net_ = caffe.Net('lenet_deploy.prototxt', 'mixed_lenet_8817_id_digit.caffemodel', caffe.TEST)

filenames = [None] * num_files
confidences = [0] * num_files
recognised_classes = [0] * num_files
exact_classes = [0] * num_files

resized_dir = join('labeled', 'resized')
classes = listdir(resized_dir)

i = 0
for class_ in classes:
    class_dir = join(resized_dir, class_)
    files = listdir(class_dir)
    for file in files:
        filenames[i] = file
        exact_classes[i] = int(class_)
        confidences[i], recognised_classes[i] = cal_conf(join(class_dir, file), net_)
        i += 1

count_errors = 0
for i in range(num_files):
    if exact_classes[i] != recognised_classes[i] and confidences[i] >= 0.9:
        count_errors += 1

print("Errors:", count_errors)
print("Min confidence:", min(confidences))
print("Max confidence:", max(confidences))

raw_data = {'file_name': filenames, 'exact_class': exact_classes,
    'recognised_class': recognised_classes,
    'confidence': confidences}

df = pd.DataFrame(raw_data, columns=['file_name', 'exact_class', 'recognised_class', 'confidence'])
df.to_csv('result_stats.csv')