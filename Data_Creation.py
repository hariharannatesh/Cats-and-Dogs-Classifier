import cv2
import numpy as np
import os
import random

from random import shuffle
from tqdm import tqdm

data_dir='Data'

img_size=28

def label_folder(folders):
    if folders== 'Cats': return [1,0]

    elif folders=='Dogs': return [0,1]


def create_train_data():
    training_data=[]
    dirs=os.listdir(data_dir)
    for folders in dirs:
        label=label_folder(folders)
        req_train_dir=os.path.join(data_dir,folders)
        for img in tqdm(os.listdir(req_train_dir)):
            path=os.path.join(req_train_dir,img)
            img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img,(img_size,img_size))
            training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('cats_dogs_data.npy',training_data)
    return training_data

train_data=create_train_data()
        
