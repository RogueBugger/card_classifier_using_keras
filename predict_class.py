#!pip install --upgrade tensorflow
#!pip install --upgrade pip setuptools wheel
#!pip install -I tensorflow
#!pip install -I keras
from google.colab import drive
#drive.mount('/content/gdrive')
import cv2 as cv
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
#https://stackoverflow.com/questions/50227925/how-to-predict-from-saved-model-in-keras

model = load_model('/content/gdrive/My Drive/dataset/model_multi_class_final.h5')

model.compile(loss='categorical_crossentropy', 
			optimizer='rmsprop', 
			metrics=['accuracy']) 
'''
test_image = image.load_img('/content/gdrive/My Drive/dataset/test.jpg', target_size = (163, 260)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
'''


#img = cv.imread('/content/gdrive/My Drive/dataset/test.jpg')
#result = model.predict(test_image)
#result = model.predict(img)
img = cv.imread('/content/gdrive/My Drive/dataset/test.jpg',0)
img=cv.resize(img,dsize=(163, 260), interpolation=cv.INTER_CUBIC)
img=np.expand_dims(img,0)
img=np.expand_dims(img,axis=3)

result = model.predict(img)
print(result)

img = cv.imread('/content/gdrive/My Drive/dataset/tes1t.jpg',0)
img=cv.resize(img,dsize=(163, 260), interpolation=cv.INTER_CUBIC)
img=np.expand_dims(img,0)
img=np.expand_dims(img,axis=3)

result = model.predict(img)
print(result)

img = cv.imread('/content/gdrive/My Drive/dataset/tes2t.jpg',0)
img=cv.resize(img,dsize=(163, 260), interpolation=cv.INTER_CUBIC)
img=np.expand_dims(img,0)
img=np.expand_dims(img,axis=3)

result = model.predict(img)
print(result)
