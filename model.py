import csv
import cv2
import numpy as np

lines = []
with open('../../../recordings/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for i, line in enumerate(lines):
    #print(i)
    source_path = line[0]
    # Need to use \, as windows path are separated by \ instead of /
    filename = source_path.split('\\')[-1]
    #print(filename)
    current_path = '../../../recordings/IMG/' + filename
    image = cv2.imread(current_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(image.shape)
    images.append(image_rgb)
    measurement = float(line[3])
    measurements.append(measurement)


aug_images, aug_measures = [], []
for img, measure in zip(images, measurements):
    aug_images.append(img)
    aug_measures.append(measure)
    aug_images.append(cv2.flip(img, 1))
    aug_measures.append(measure * -1.0)

#X_train = np.array(images)
#y_train = np.array(measurements)

X_train = np.array(aug_images)
y_train = np.array(aug_measures)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

'''
model = Sequential()
model.add(Lambda(lambda x: (x/255.0 - 0.5), input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))
'''

'''
model = Sequential()
model.add(Lambda(lambda x: (x/255.0 - 0.5), input_shape=(160, 320, 3)))
# Crop the top 60 pixels and bottom 20 pixels
model.add(Cropping2D(cropping=((60, 20), (0, 0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
'''

model = Sequential()
model.add(Lambda(lambda x: (x/255.0 - 0.5), input_shape=(160, 320, 3)))
# Crop the top 60 pixels and bottom 20 pixels
model.add(Cropping2D(cropping=((60, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch=5)

model.save('model.h5')