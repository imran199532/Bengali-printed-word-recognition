import  os
import cv2
import numpy as np
from os.path import basename
import sys
import shutil

path='C:/Users/Badhon/PycharmProjects/imran/labels'

count=0
breakPoint=73


for fileName in os.listdir(path):



    base=os.path.splitext(fileName)[0]
    filePath='C:/Users/Badhon/PycharmProjects/imran/labels/%s.txt' % base
    with open(filePath) as f:
        txt=f.readlines()

        newF = open('modified_labels/%s.txt' % base, "w+")
        for line in txt:
            #print(line)
            val=int(line)
            if val>breakPoint:
                temp=str(breakPoint+1)
                newF.write(temp)
            else:
                newF.write('%s'%line)




        newF.close()


print(count)
