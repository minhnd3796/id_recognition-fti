import cv2
from os.path import join
from os import listdir
from recognise_dob import crop_dob

input_dir = 'date_of_birth_valid'
output_dir = 'date_of_birth_valid_RESULTS'

input_file = listdir(input_dir)
for file in input_file:
    print(file)
    out_dd, out_mm, out_yyyy = crop_dob(cv2.imread(join(input_dir, file), 0))
    cv2.imwrite(join(output_dir, 'dd', file), out_dd)
    cv2.imwrite(join(output_dir, 'mm', file), out_mm)
    cv2.imwrite(join(output_dir, 'yyyy', file), out_yyyy)