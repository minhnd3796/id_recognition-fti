from os.path import join, isfile
from os import listdir

src_dir_ = 'original_size'

def count_files(src_dir):
    dirs = [folder for folder in listdir(src_dir) if not isfile(join(src_dir, folder))]
    count = 0
    for dir_ in dirs:
        dir_ = join(src_dir, dir_)
        classes = listdir(dir_)
        for class_ in classes:
            file_dir = join(dir_, class_)
            # print(file_dir)
            files = listdir(file_dir)
            # print(files)
            count = count + len(files)
    return count

print(count_files(src_dir_))
