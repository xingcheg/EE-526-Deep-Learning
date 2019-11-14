from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense


batch_size = 500
L = 10
epochs = 20

# load and transform data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train/255
x_test = x_test/255

# convert class vectors to one hot matrix
y_train = keras.utils.to_categorical(y_train, L)
y_test = keras.utils.to_categorical(y_test, L)

# model structure
model = Sequential()
model.add(Dense(L, activation='softmax', input_shape=(784,)))

# model run
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.01)

# model evaluation
score_train = model.evaluate(x_train, y_train, verbose=0)
score_test = model.evaluate(x_test, y_test, verbose=0)

print('Training accuracy:', score_train[1])
print('Testing accuracy:', score_test[1])
