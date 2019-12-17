from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Activation, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
#from google.colab import drive
#drive.mount('/content/gdrive')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('./gdrive/My Drive/dataset/atm/IMG_20191217_183620_1.jpg')
imgplot = plt.imshow(img)
plt.show()
