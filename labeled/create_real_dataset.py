from os import listdir
from os.path import join
import cv2

train_images = []
train_labels = []
test_images = []
test_labels = []

train_dir = join('resized')
classes = listdir(train_dir)
classes.sort()
for class_ in classes:
    path_to_class = join(train_dir, class_)
    files = listdir(path_to_class)
    for file in files:
        path_to_file = join(path_to_class, file)
        train_images.append(path_to_file)
        train_labels.append(int(class_))

''' test_dir = join("RealData", "test")
classes = listdir(test_dir)
classes.sort()
for class_ in classes:
    path_to_class = join(test_dir, class_)
    files = listdir(path_to_class)
    for file in files:
        path_to_file = join(path_to_class, file)
        test_images.append(path_to_file)
        test_labels.append(int(class_)) '''

num_train = len(train_labels)
# num_test = len(test_labels)
LABEL_MAGIC = 2049
IMAGE_MAGIC = 2051
NUM_ROWS = 28
NUM_COLS = 28

f_train_images = open('train-images-idx3-ubyte.overwritten', 'r+b')
# f_train_images.seek(4, 0)
# f_train_images.write((num_train + 60000).to_bytes(4, byteorder="big", signed=False))
# f_train_images.seek(0, 2)
f_train_images.seek(16, 0)
for img_path in train_images:
    img = cv2.imread(img_path, 0)
    img = img.reshape((784,))
    for i in range(784):
        f_train_images.write(int((img[i])).to_bytes(1, byteorder="big", signed=False))
f_train_images.close()

f_train_labels = open('train-labels-idx1-ubyte.overwritten', 'r+b')
# f_train_labels.seek(4, 0)
# f_train_labels.write((num_train + 60000).to_bytes(4, byteorder="big", signed=False))
# f_train_labels.seek(0, 2)
f_train_labels.seek(8, 0)
f_train_labels
for label in train_labels:
    f_train_labels.write((label).to_bytes(1, byteorder="big", signed=False))
f_train_labels.close()

''' f_test_images = open('t10k-images-idx3-ubyte.appended', 'r+b')
f_test_images.seek(4, 0)
f_test_images.write((num_test + 10000).to_bytes(4, byteorder="big", signed=False))
f_test_images.seek(0, 2)
for img_path in test_images:
    img = cv2.imread(img_path, 0)
    img = img.reshape((784,))
    for i in range(784):
        f_test_images.write(int((img[i])).to_bytes(1, byteorder="big", signed=False))
f_test_images.close()

f_test_labels = open('t10k-labels-idx1-ubyte.appended', 'r+b')
f_test_labels.seek(4, 0)
f_test_labels.write((num_test + 10000).to_bytes(4, byteorder="big", signed=False))
f_test_labels.seek(0, 2)
for label in test_labels:
    f_test_labels.write((label).to_bytes(1, byteorder="big", signed=False))
f_test_labels.close() '''
