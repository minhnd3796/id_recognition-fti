#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 11:21:39 2018

@author: minhnd
"""

#%%
from os import listdir, rename
from os.path import join
#%%
data_dir = join("ExtraData", "digits_padded")
folders = listdir(data_dir)
folders.sort()
#%%
for folder in folders:
    files = listdir(join(data_dir, folder))
    i = 0
    for file in files:
        pwd = join(data_dir, folder)
        i = i + 1
        rename(join(pwd, file), join(pwd, str(folder) + '_' + str(i) + '.png'))
#%%