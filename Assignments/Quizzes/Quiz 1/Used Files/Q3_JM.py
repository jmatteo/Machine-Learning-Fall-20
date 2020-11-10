'''

Q3 Inception Module for CIFAR-10 dataset

'''
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.layers import Input
from keras.utils import np_utils
from keras.datasets import cifar10

epochs = 100

# Get the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Get the data ready
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Create input
input_img = Input(shape = (32, 32, 3))


# Create Volumes for the Inception module
volume_1 = Conv2D(96, (1,1), padding='same', activation='relu')(input_img)

volume_2 = Conv2D(96, (1,1), padding='same', activation='relu')(input_img)
volume_2 = Conv2D(128, (3,3), padding='same', activation='relu')(volume_2)

volume_3 = Conv2D(32, (1,1), padding='same', activation='relu')(input_img)
volume_3 = Conv2D(64, (5,5), padding='same', activation='relu')(volume_3)

volume_4 = MaxPooling2D((2,2), strides=(1,1), padding='same')(input_img)
volume_4 = Conv2D(32, (1,1), padding='same', activation='relu')(volume_4)

volume_5 = Conv2D(128, (2,2), padding='same', activation='relu')(input_img)
volume_5 = Conv2D(256, (5,5), padding='same', activation='relu')(volume_5)

volume_6 = Conv2D(32, (1,1), padding='same', activation='relu')(input_img)
volume_6 = Conv2D(64, (10,10), padding='same', activation='relu')(volume_6)

volume_7 = MaxPooling2D((2,2), strides=(1,1), padding='same')(volume_3)
volume_7 = Conv2D(32, (4,4), padding='same', activation='relu')(volume_7)

volume_8 = MaxPooling2D((2,2), strides=(1,1), padding='same')(volume_1)
volume_8 = Conv2D(32, (4,4), padding='same', activation='relu')(volume_8)

volume_9 = MaxPooling2D((2,2), strides=(1,1), padding='same')(volume_5)
volume_9 = Conv2D(32, (4,4), padding='same', activation='relu')(volume_9)

volume_10 = MaxPooling2D((2,2), strides=(1,1), padding='same')(volume_5)
volume_10 = Conv2D(32, (4,4), padding='same', activation='relu')(volume_10)

volume_11 = MaxPooling2D((2,2), strides=(1,1), padding='same')(volume_10)
volume_11 = Conv2D(32, (4,4), padding='same', activation='relu')(volume_11)

volume_12 = MaxPooling2D((2,2), strides=(1,1), padding='same')(volume_7)
volume_12 = Conv2D(32, (4,4), padding='same', activation='relu')(volume_12)

# Concatenate all volumes of the Inception module
inception_module = keras.layers.concatenate([volume_1, volume_2, volume_3,
                                             volume_4, volume_5, volume_6,
                                             volume_7, volume_8, volume_9,
                                             volume_10, volume_11, volume_12], axis = 3)
output = Flatten()(inception_module)
out    = Dense(10, activation='softmax')(output)


model = Model(inputs = input_img, outputs = out)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=512)


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
