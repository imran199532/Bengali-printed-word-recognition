import  os
import cv2
import numpy as np
from os.path import basename
import sys
import shutil

path='C:/Users/Badhon/PycharmProjects/imran/label_word_id'
path1='C:/Users/Badhon/PycharmProjects/imran/train'

count=0




for fileName in os.listdir(path):

    count+=1
    continue


    base=os.path.splitext(fileName)[0]
    filePath='C:/Users/Badhon/PycharmProjects/imran/label_word_id/%s.txt' % base
    with open(filePath) as f:
        txt=f.readline()


        size=sys.getsizeof(txt)
        count+=1
        if size==49:
            continue

        print(base)
        shutil.copy(filePath,'labels')

        filePath='C:/Users/Badhon/PycharmProjects/imran/literature_image_words/%s.jpg'%base
        shutil.copy(filePath,'train')
       # count+=1




print(count)
