from google.colab import drive
drive.mount('/content/gdrive')
import cv2 as cv
import os
import numpy as np

os.chdir('/content/gdrive/My Drive/dataset/New/validation/')


for j in os.listdir('/content/gdrive/My Drive/dataset/validatin/'):
  for i in (os.listdir('/content/gdrive/My Drive/dataset/validatin/'+j)):
    img = cv.imread('/content/gdrive/My Drive/dataset/validatin/'+j+'/'+i)
    img = cv.resize(img,dsize = (260, 163),interpolation=cv.INTER_CUBIC)  
    cv.imwrite(i, img)
