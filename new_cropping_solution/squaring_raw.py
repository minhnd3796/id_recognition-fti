from recognise_id_number import square_padding
from os import listdir
from os.path import join
import cv2
import sys

raw_dir = sys.argv[1]

images_filenames = listdir(raw_dir)

for file in images_filenames:
    file_path = join(raw_dir, file)
    img = cv2.imread(file_path, 0)
    squared_img = square_padding(img)
    cv2.imwrite(file_path, squared_img)
