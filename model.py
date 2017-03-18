import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, \
    Activation, Lambda, Cropping2D

# import the data
lines = []
with open('./data5/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
# data is contained in image and measurements
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# the following codes are used to get more data,
# but I'm not going to using them for now
'''
for line in lines:
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurement += 0.18
    measurements.append(measurement)

for line in lines:
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurement -= 0.22
    measurements.append(measurement)
'''
'''
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
'''

X_train = np.array(images)
y_train = np.array(measurements)

# the architecture of my model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((20, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(3, 3), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(3, 3), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

# save the model
model.save('model.h5')
