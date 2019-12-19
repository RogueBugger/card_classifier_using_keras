#!pip install --upgrade tensorflow
#!pip install --upgrade pip setuptools wheel
#!pip install -I tensorflow
#!pip install -I keras
from google.colab import drive
#drive.mount('/content/gdrive')
import cv2 as cv
import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import json

model = load_model('/content/gdrive/My Drive/dataset/model_multi_class_final.h5')

model.compile(loss='categorical_crossentropy', 
			optimizer='rmsprop', 
			metrics=['accuracy']) 

f = open('/content/gdrive/My Drive/dataset/class_indices.json')
data = json.load(f)

data = dict(map(reversed, data.items()))

def predict_class(path):
	img = cv.imread(path,0)
	img=cv.resize(img,dsize=(163, 260), interpolation=cv.INTER_CUBIC)
	img=np.expand_dims(img,0)
	img=np.expand_dims(img,axis=3)
	result = model.predict(img)
	print(result)
	arr = result.argmax(axis=-1)
	print(data[arr[0]])

predict_class('/content/gdrive/My Drive/dataset/test.jpg')
predict_class('/content/gdrive/My Drive/dataset/tes1t.jpg')
predict_class('/content/gdrive/My Drive/dataset/tes2t.jpg')
