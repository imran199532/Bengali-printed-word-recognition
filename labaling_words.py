import  os
import cv2
import numpy as np
from os.path import basename

path='C:/Users/Badhon/PycharmProjects/imran/literature_image_words'
count=0
done=0

for fileName in os.listdir('C:/Users/Badhon/PycharmProjects/imran/literature_labels'):
    done+=1

print(done)



for fileName in os.listdir(path):
    if count<done:
        count+=1
        continue
    img = cv2.imread('C:/Users/Badhon/PycharmProjects/imran/literature_image_words/%s'% fileName, 0)
    if img is None:
        count+=1
        continue
    count += 1

    base=os.path.splitext(fileName)[0]
    print(base)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    f = open('literature_labels/%s.txt' % base, "w+")
    f.close()
    file = "notepad.exe C:/Users/Badhon/PycharmProjects/imran/literature_labels/%s.txt" % base
    os.system(file)



print(count)
