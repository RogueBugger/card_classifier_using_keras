#!pip install --upgrade tensorflow
#!pip install --upgrade pip setuptools wheel
#!pip install -I tensorflow
#!pip install -I keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Activation, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
from google.colab import drive
#drive.mount('/content/gdrive')
import json
#mport matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#img=mpimg.imread('./gdrive/My Drive/dataset/atm/IMG_20191217_183620_1.jpg')
#imgplot = plt.imshow(img)
#plt.show()


#Note = image need to be rescaled to low resolution for training.
image_width = 260
image_height = 163
img_shape = (image_width, image_height, 1)




model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = img_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', 
			optimizer='rmsprop', 
			metrics=['accuracy']) 

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/content/gdrive/My Drive/dataset/New/Train/',
        target_size=(image_width, image_height),
        color_mode='grayscale',
        batch_size=45,
        class_mode='categorical')


jfile = open('class_indices.txt','w',encoding='utf-8')
json.dump(train_generator.class_indices, jfile)

validation_generator = test_datagen.flow_from_directory(
        '/content/gdrive/My Drive/dataset/New/validation/',
        target_size=(image_width, image_height),
        batch_size=45,
        color_mode='grayscale',
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=32,
        epochs=12,
        validation_data=validation_generator,
        validation_steps=32)


model.save('model_multi_class_final.h5')
