import zipfile
import numpy as np
import os

def un_zip(file_name):
    """unzip zip file"""
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(file_name + "_files"):
        pass
    else:
        os.mkdir(file_name + "_files")
    for names in zip_file.namelist():
        zip_file.extract(names,file_name + "_files/")
    zip_file.close()

# un_zip('train_x.zip')
data = np.load('train_x.zip_files/train_x.npy')
train_y = np.array(np.load('train_y.npy'))
print(type(data))
print(data.shape)
print(train_y)
print(train_y.shape)
