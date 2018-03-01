import cv2
from os import listdir, rename
from os.path import join

data_dir = join("ExtraData", "digits_padded")
folders = listdir(data_dir)
folders.sort()

for folder in folders:
    files = listdir(join(data_dir, folder))
    # i = 0
    for file in files:
        pwd = join(data_dir, folder)
        # i = i + 1
        # rename(join(pwd, file), join(pwd, str(folder) + '_' + str(i) + '.png'))
        full_filename = join(pwd, file)
        img = cv2.imread(join(pwd, file), 0)
        img_resized = cv2.resize(img, (28, 28))
        cv2.imwrite(full_filename, img_resized)
