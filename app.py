import numpy as np
import math
import tensorflow as tf
from test import test_data
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
trainning = False
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #convert class vectors to binary class matrics
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
   


    # x_test = np.random.nomal(x_test)

    x_train = x_train / 255
    x_test = x_test / 255
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

model = Sequential()
model.add(Dense(input_dim=28*28, units=689, activation='tanh')) #sigmoid to relu
for i in range(3):
    model.add(Dense(units=400,activation='relu')) #sigmoid to relu
model.add(Dense(units=10,activation='softmax'))
# mse to categorical_crossentropy
if trainning:
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    # start trainning 
    model.fit(x_train, y_train, batch_size=1200, epochs=20)


    #save weights
    model.save_weights('weights.h5',overwrite=True)


    #validation
    result = model.evaluate(x_train, y_train, batch_size=240)
    print('\nTrain Acc' , result[1])

    result = model.evaluate(x_test, y_test, batch_size=240)
    print('\nTest Acc' , result[1])

else:
    #load weights
    model.load_weights('weights.h5')


    # predit
    files = [ '0.BMP', '1.BMP', '2.BMP', '3.BMP', '4.BMP', '5.BMP', '6.BMP', '7.BMP', '8.BMP', '9.BMP']
    for i in range(10):

        array = list(model.predict( test_data(files[i]))[0])
        # print(array)
        print ( array.index( max(array) ) )






