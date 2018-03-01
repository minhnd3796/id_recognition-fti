import cv2
from os import listdir, rename, makedirs
from os.path import join, exists

original_size_dir = 'original_size'
folders = listdir(original_size_dir)
# folders.sort()

for folder in folders:
    folder_dir = join(original_size_dir, folder)
    classes = listdir(folder_dir)
    # i = 0
    for class_ in classes:
        pwd = join(folder_dir, class_)
        files = listdir(pwd)
        for file in files:
            # i = i + 1
            # rename(join(pwd, file), join(pwd, str(folder) + '_' + str(i) + '.png'))
            full_filename = join(pwd, file)
            img = cv2.imread(full_filename, 0)
            img_resized = cv2.resize(img, (28, 28))
            out_dir = join('resized', class_)
            if not exists(out_dir):
                makedirs(out_dir)
            cv2.imwrite(join(out_dir, file), img_resized)
