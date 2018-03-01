import os
import shutil

RAW_SRC_DIR = '/home/minhnd/Desktop/MNIST/new digits/raw_data/id_number_combined'
BOXED_DST_DIR = '/home/minhnd/Desktop/mixed-MNIST-case-study/idnumber_to_text/number_id_test_with_boxes'
RAW_DST_DIR = '/home/minhnd/Desktop/mixed-MNIST-case-study/idnumber_to_text/number_id_test'

boxed_sub_dirs = os.listdir(BOXED_DST_DIR)
for dir_ in boxed_sub_dirs:
    boxed_path = os.path.join(BOXED_DST_DIR, dir_)
    raw_path = os.path.join(RAW_DST_DIR, dir_)
    filenames = os.listdir(boxed_path)
    for filename in filenames:
        boxed_file_path = os.path.join(boxed_path, filename)
        raw_file_path = os.path.join(raw_path, filename)
        shutil.copyfile(os.path.join(RAW_SRC_DIR, filename), raw_file_path)
