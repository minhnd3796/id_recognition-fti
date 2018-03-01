from shutil import move
import sys
from os import listdir
from os.path import join

number_class = sys.argv[1]
num_moved = 7000

in_dir = join('labelled', str(number_class))
out_dir = join('seventy_thousand_set', str(number_class))

files = listdir(in_dir)
for i in range(num_moved):
    move(join(in_dir, files[i]), join(out_dir, files[i]))
    print(i+1, '/', num_moved, 'moved...')