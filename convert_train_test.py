import  os
import cv2
import numpy as np
from os.path import basename
import sys
import shutil

path='C:/Users/Badhon/PycharmProjects/imran/label_word_id'

count=0


for fileName in os.listdir('modified_labels'):

    count+=1


train_size=int(count*.80)
print(count,train_size)

cnt=0
i=6
fold_10=count/10.0



for fileName in os.listdir('test'):
    base = os.path.splitext(fileName)[0]
   # filePath = 'C:/Users/Badhon/PycharmProjects/imran/modified_labels/%s.txt' % base



    filePath = 'C:/Users/Badhon/PycharmProjects/imran/test/%s.jpg' % base
    shutil.copy(filePath, 'train')
    os.remove(filePath)

for fileName in os.listdir('train'):
    cnt+=1
    if cnt<fold_10*i or cnt>=fold_10*(i+1):
        continue

    base = os.path.splitext(fileName)[0]
    filePath = 'C:/Users/Badhon/PycharmProjects/imran/modified_labels/%s.txt' % base



    filePath = 'C:/Users/Badhon/PycharmProjects/imran/train/%s.jpg' % base
    shutil.copy(filePath, 'test')
    os.remove(filePath)





