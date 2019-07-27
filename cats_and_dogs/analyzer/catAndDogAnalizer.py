from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display


# BATCH_SIZE
BATCH_SIZE = 32

# Handling data
# First we will set data augmentation: (Data augmentation is used to extend our data set)
# ImageDataGenerator:
# Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).
# rescale:
# is a value by which we will multiply the data before any other processing.
# Our original images consist in RGB coefficients in the 0-255,
# but such values would be too high for our models to process (given a typical learning rate),
# so we target values between 0 and 1 instead by scaling with a 1/255. factor.
#
# shear_range:
# is for randomly applying shearing transformations
#
# zoom_range:
# is for randomly zooming inside pictures
#
# horizontal_flip:
# is for randomly flipping half of the images horizontally
# --relevant when there are no assumptions of horizontal assymetry (e.g. real-world pictures).
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

training_set = train_datagen.flow_from_directory('../dataset/training_set', target_size=(64, 64),
                                                 batch_size=BATCH_SIZE, class_mode='binary')

test_set = train_datagen.flow_from_directory('../dataset/test_set', target_size=(64, 64),
                                                 batch_size=BATCH_SIZE, class_mode='binary')

# Init CNN
classifier = Sequential()

# Adding layers
# Convolution2D:
# Filters: Feature maps to sample against our image (pixel map)
# kernel size: in our case RGB pic = 3
# input_shape=(64, 64, 3) 64 * 64 image with colors
# relu => F(x) = MAX(0,x)
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Max pooling
# Choosing the grater feature out of the convolutions feature maps for each 2X2 Block
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Flatten (prepare for dense)
classifier.add(Flatten())

# Dense layers
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='relu'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit
classifier.fit_generator(training_set, steps_per_epoch=8000, epochs=10, validation_data=test_set, validation_steps=800)
