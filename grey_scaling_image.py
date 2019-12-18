from google.colab import drive
#drive.mount('/content/gdrive')
import cv2 as cv
import os
import numpy as np

img = cv.imread('/content/gdrive/My Drive/dataset/drivee.jpg')
img = cv.resize(img, dsize = (260, 163), interpolation=cv.INTER_CUBIC)
os.chdir('/content/gdrive/My Drive/dataset')
cv.imwrite('tes2t.jpg', img)
print(img.shape)
