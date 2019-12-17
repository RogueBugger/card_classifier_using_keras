#!pip install --upgrade tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Activation, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
from google.colab import drive
drive.mount('/content/gdrive')
#mport matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#img=mpimg.imread('./gdrive/My Drive/dataset/atm/IMG_20191217_183620_1.jpg')
#imgplot = plt.imshow(img)
#plt.show()

image_width = 4000
image_height = 3000
img_shape = (image_width, image_height, 3)




model = Sequential()
model.add(Conv2D(32, (2,2), input_shape = img_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy', 
			optimizer='rmsprop', 
			metrics=['accuracy']) 

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './gdrive/My Drive/dataset/train/',
        target_size=(image_width, image_height),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        './gdrive/My Drive/dataset/validation/',
        target_size=(image_width, image_height),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=80,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=40)


model.save_weights('model_saved.h5')
