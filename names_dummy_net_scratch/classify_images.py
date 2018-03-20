from os import listdir, makedirs
from os.path import exists, join
from shutil import move

input_dir = 'AllProvince'

with open('label.txt') as f:
    lines = f.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].split()

for line in lines:
    output_dir = join(input_dir, line[1].upper())
    if not exists(output_dir):
        makedirs(output_dir)
    move(join(input_dir, line[0]), join(output_dir, line[0]))