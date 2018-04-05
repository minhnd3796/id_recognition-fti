from pandas import read_csv
import numpy as np
from os import listdir, mkdir
from ntpath import basename
from recognise_id_number import idToStr, crop_id
from shutil import copyfile
from os.path import exists

dataset = read_csv('table.csv')
_id = dataset.iloc[:, 1].values
dob = dataset.iloc[:, 2].values
file = dataset.iloc[:, 4].values

# print(type(str(_id[0])))
for i in range(len(file)):
    file[i] = basename(file[i])[:-4]

file_from_dir = listdir('../id_number_combined')

available_pos_from_dir = []
available_pos_from_table = []
for i in range(len(file)):
    filename = file[i] + '.jpgcardnumber.png'
    if filename in file_from_dir:
        pos_from_dir = file_from_dir.index(filename)
        available_pos_from_dir.append(pos_from_dir)
        available_pos_from_table.append(i)

file_from_dir = np.array(file_from_dir)[np.array(available_pos_from_dir)]
file = file[np.array(available_pos_from_table)]
_id = _id[np.array(available_pos_from_table)]
dob = dob[np.array(available_pos_from_table)]

correct_id = 0
total = len(file)
for i in range(total):
    pred_id = idToStr(crop_id('../id_number_combined/' + file_from_dir[i]), '../lenet_deploy.prototxt', '../lenet_iter_100000.caffemodel')
    if pred_id != '':
        for j in range(len(pred_id)):
            if pred_id[j] != '0':
                break
        pred_id = pred_id[j:]
    matches = str(_id[i]) == pred_id
    if matches:
        correct_id += 1
    else:
        if not exists('id_error_images/'):
            mkdir('id_error_images/')
        copyfile('../id_number_combined/' + file_from_dir[i], 'id_error_images/' + file_from_dir[i])
    print(">> Testing", file, matches)
print('Correct:', correct_id)
print('Total:', total)
print('Accuracy:', correct_id * 1.0 / total)