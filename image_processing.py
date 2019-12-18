from google.colab import drive
#drive.mount('/content/gdrive')
import cv2 as cv
import os
import numpy as np


#image_path = '/content/gdrive/My Drive/dataset/train/atm/IMG_20191217_183620_1.jpg'
#img = cv.imread(image_path,1)
#img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#print(os.getcwd())
#print(img.shape)
#img = cv.resize(img,dsize = (256, 256),interpolation=cv.INTER_CUBIC)
#os.chdir()
#print(img.shape)
#cv.imwrite('test.jpg', img)
for j in (os.listdir('/content/gdrive/My Drive/dataset/train/')):
  print(j)
  for i in (os.listdir('/content/gdrive/My Drive/dataset/train/'+j)):
    #print('/content/gdrive/My Drive/dataset/train/'+j+'/'+i)
    img = cv.imread('/content/gdrive/My Drive/dataset/train/'+j+'/'+i)
    img = cv.resize(img,dsize = (256, 256),interpolation=cv.INTER_CUBIC)  
    print(img.shape)
  '''
  for i in (os.listdir('/content/gdrive/My Drive/dataset/train/'+j)):
    img = cv.imread('/content/gdrive/My Drive/dataset/train/'+j+i)
    img = cv.resize(img,dsize = (256, 256),interpolation=cv.INTER_CUBIC)  
    cv.imwrite(i, img)
  '''
