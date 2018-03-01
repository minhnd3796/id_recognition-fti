import cv2
import os

dob_files = os.listdir('dob_bin')
for dob_file in dob_files:
    img = cv2.imread(os.path.join('dob_bin', dob_file))
    img = 255 - img
    cv2.imwrite(os.path.join('dob_bin_inverted', dob_file), img)